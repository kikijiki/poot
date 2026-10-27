//! The `#[kernel]` attribute (a kernel is ordinary Rust, lowered from real rustc MIR).
//!
//! Two-phase, selected by this crate's own `device-pass` Cargo feature (card 537, ADR-0104 decision 5,
//! SC-002: the macro takes no environment input - a proc macro that read `std::env::var` at expansion time
//! was a mode-selecting env read in library code; the choice is now which of two proc-macro artifacts the
//! build links in, a typed, build-time decision, not a runtime one):
//!
//! - **Device pass** (this crate built `--features device-pass`, the artifact `pootc` extern-links for
//!   kernel extraction): rename the function into poot's reserved namespace (`__poot_kernel_<name>`) and
//!   make it `pub` so it survives to MIR, where `pootc` discovers it by name (`naming::is_kernel`) and
//!   imports + compiles its Stable MIR. The body stays ordinary Rust that rustc type/borrow-checks - the
//!   macro does no codegen, defines no DSL.
//! - **Host pass** (this crate's default build, what an ordinary `cargo build` links): emit a typed launch
//!   wrapper with the original name and signature plus a leading `&Context`, that dispatches the kernel via
//!   `poot_wgpu_runtime::Context::launch`. The body is not re-emitted, so kernel bodies need
//!   not be host-compatible. The wrapper builds its `CompiledKernel` with `poot_codegen::kernel_handle`
//!   from the kernel-IR `Body` the device pass compiled (card 608), over the artifact
//!   `poot_codegen::KernelCache::load_or_compile` returns for that same `Body`:
//!   `poot-codegen` stays the only place that constructs a `CompiledKernel`, so a `#[kernel]`'s schema and
//!   trap flag always come from the verified `Body`, never guessed from the Rust signature, and the code
//!   always comes from that `Body`'s own compile. The wrapper
//!   looks the `Body` up by name via `crate::poot_kernel_body` (the
//!   host crate `include!`s the pootc-generated `poot_cache.rs` at its crate root, and so needs
//!   `poot-codegen`, `poot-kernel-ir` and `serde_json` as dependencies alongside `poot-wgpu-runtime`). The
//!   kernel's writable `&mut [T]` output must be its last parameter (the binding order).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
#[cfg(any(test, not(feature = "device-pass")))]
use syn::{FnArg, Pat};
use syn::{Ident, ItemFn, LitInt, Token, Type, parse_macro_input};

/// What `#[kernel(...)]` declares of the launch, so `pootc` reads a declaration instead of inferring one:
///
/// - `workgroup(x)`, `workgroup(x, y)` or `workgroup(x, y, z)`: the workgroup shape in lanes (the missing axes
///   are 1);
/// - `lds(T, len)`, once per workgroup-local array, in array-id order: `len` elements of `T` (`f32`, `u32` or
///   `i32`).
///
/// A kernel that reads a lane or workgroup index, calls the barrier or touches an LDS array must declare its
/// workgroup shape.
#[derive(Default)]
struct LaunchDecl {
    workgroup: Option<[usize; 3]>,
    lds: Vec<(Type, usize)>,
}

impl Parse for LaunchDecl {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut decl = LaunchDecl::default();
        let items = Punctuated::<LaunchItem, Token![,]>::parse_terminated(input)?;
        for item in items {
            match item {
                LaunchItem::Workgroup(shape, span) => {
                    if decl.workgroup.replace(shape).is_some() {
                        return Err(syn::Error::new(span, "`workgroup(..)` is declared twice"));
                    }
                }
                LaunchItem::Lds(element, len) => decl.lds.push((*element, len)),
            }
        }
        Ok(decl)
    }
}

enum LaunchItem {
    Workgroup([usize; 3], proc_macro2::Span),
    Lds(Box<Type>, usize),
}

impl Parse for LaunchItem {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        let args;
        syn::parenthesized!(args in input);
        match name.to_string().as_str() {
            "workgroup" => {
                let extents = Punctuated::<LitInt, Token![,]>::parse_terminated(&args)?;
                if extents.is_empty() || extents.len() > 3 {
                    return Err(syn::Error::new(
                        name.span(),
                        "`workgroup(..)` takes 1 to 3 lane counts",
                    ));
                }
                let mut shape = [1usize; 3];
                for (axis, extent) in extents.iter().enumerate() {
                    shape[axis] = extent.base10_parse()?;
                    if shape[axis] == 0 {
                        return Err(syn::Error::new(
                            extent.span(),
                            "a workgroup extent is at least 1",
                        ));
                    }
                }
                Ok(LaunchItem::Workgroup(shape, name.span()))
            }
            "lds" => {
                let element: Type = args.parse()?;
                args.parse::<Token![,]>()?;
                let len: LitInt = args.parse()?;
                Ok(LaunchItem::Lds(Box::new(element), len.base10_parse()?))
            }
            _ => Err(syn::Error::new(
                name.span(),
                "unknown #[kernel(..)] argument: expected `workgroup(..)` or `lds(T, len)`",
            )),
        }
    }
}

/// Marks a function as a poot kernel. Device pass: renames it into the reserved kernel namespace and makes it
/// `pub` so `pootc` finds it in MIR (see the crate doc).
#[cfg(feature = "device-pass")]
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let decl = parse_macro_input!(attr as LaunchDecl);
    let func = parse_macro_input!(item as ItemFn);
    device_fn(func, &decl).into()
}

/// Marks a function as a poot kernel. Host pass: replaces it with a typed launch wrapper of the same name
/// that dispatches the kernel `pootc` compiled (see the crate doc).
#[cfg(not(feature = "device-pass"))]
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    // The host launches the `Body` the device pass compiled, which carries the declared shape: the declaration is
    // parsed here to reject a malformed one, and read by `pootc`.
    parse_macro_input!(attr as LaunchDecl);
    let func = parse_macro_input!(item as ItemFn);
    host_wrapper(&func)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Device pass: rename to the reserved namespace + make `pub` so the kernel survives to MIR for pootc, and
/// record the declared launch as `poot_kernel_intrinsics::declare_*` calls at the head of the body: `pootc`
/// reads their constants and drops the calls, so the declaration is data in the MIR, resolved by the callee's
/// DefId like every other intrinsic.
/// Compiled outside `--features device-pass` too, under `cfg(test)`: SC-002's
/// `device_fn_expansion_is_deterministic` calls it directly to prove it takes no environment input,
/// without needing a second build of this crate.
#[cfg(any(test, feature = "device-pass"))]
fn device_fn(mut func: ItemFn, decl: &LaunchDecl) -> TokenStream2 {
    let mangled = poot_kernel_ir::naming::mangle(&func.sig.ident.to_string());
    func.sig.ident = Ident::new(&mangled, func.sig.ident.span());
    func.vis = syn::parse_quote!(pub);
    let mut declarations: Vec<syn::Stmt> = Vec::new();
    if let Some([x, y, z]) = decl.workgroup {
        declarations.push(syn::parse_quote! {
            ::poot_kernel_intrinsics::declare_workgroup(#x, #y, #z);
        });
    }
    for (element, len) in &decl.lds {
        declarations.push(syn::parse_quote! {
            ::poot_kernel_intrinsics::declare_lds::<#element>(#len);
        });
    }
    func.block.stmts.splice(0..0, declarations);
    quote!(#func)
}

/// Whether a `#[kernel]` parameter is a writable output (`&mut [T]`, the schema's `ArgAccess::Write`) or a
/// read-only input (`&[T]`): only the reference's mutability, read here to pick the wrapper's binding
/// order. The element type `T` no longer needs reading here (card 608): the
/// schema comes from the compiled `Body`, not from Rust syntax, so an unsupported or mismatched element
/// type is instead a normal type error where the wrapper calls `Context::launch` (its parameters are
/// concretely typed `&[f32]`/`&mut [f32]`).
#[cfg(any(test, not(feature = "device-pass")))]
fn is_mut_ref(ty: &Type) -> syn::Result<bool> {
    match ty {
        Type::Reference(r) => Ok(r.mutability.is_some()),
        _ => Err(syn::Error::new_spanned(
            ty,
            "a #[kernel] param must be &[T] or &mut [T]",
        )),
    }
}

/// Host pass: a typed `fn name(ctx, <params>) -> Result<(), RuntimeError>` that dispatches the cached
/// SPIR-V. Read-only `&[T]` params bind first (in order); the single `&mut [T]` (last) is the output.
/// Compiled under `--features device-pass` too, under `cfg(test)`: SC-002's
/// `host_wrapper_expansion_is_independent_of_poot_kernel_build_env` calls it directly.
#[cfg(any(test, not(feature = "device-pass")))]
fn host_wrapper(func: &ItemFn) -> syn::Result<TokenStream2> {
    let name = &func.sig.ident;
    let name_str = name.to_string();
    let mut params = Vec::new();
    let mut inputs = Vec::new();
    let mut output = None;
    let arg_count = func.sig.inputs.len();
    for (i, arg) in func.sig.inputs.iter().enumerate() {
        let FnArg::Typed(pt) = arg else {
            return Err(syn::Error::new_spanned(arg, "a #[kernel] takes no `self`"));
        };
        let Pat::Ident(pi) = &*pt.pat else {
            return Err(syn::Error::new_spanned(
                &pt.pat,
                "kernel params must be plain identifiers",
            ));
        };
        let ident = pi.ident.clone();
        params.push(quote!(#pt));
        let is_out = is_mut_ref(&pt.ty)?;
        if is_out {
            if output.is_some() {
                return Err(syn::Error::new_spanned(
                    pt,
                    "a #[kernel] has exactly one `&mut` output",
                ));
            }
            // The device binds params by declaration order; the host wrapper calls
            // `ctx.launch(kernel, &inputs, output)` (inputs first, output last). A mid-position `&mut`
            // (`fn k(a, out: &mut, b)`) would compile silently with mismatched bindings and corrupt an
            // input, so reject it here.
            if i != arg_count - 1 {
                return Err(syn::Error::new_spanned(
                    pt,
                    "a #[kernel]'s `&mut` output must be its LAST parameter (the device binds by \
                     declaration order; the host wrapper binds inputs-then-output)",
                ));
            }
            output = Some(ident);
        } else {
            inputs.push(ident);
        }
    }
    let output = output.ok_or_else(|| {
        syn::Error::new_spanned(
            &func.sig,
            "a #[kernel] needs one `&mut [T]` output parameter (last)",
        )
    })?;
    Ok(quote! {
        pub fn #name(
            ctx: &::poot_wgpu_runtime::Context,
            #(#params),*
        ) -> ::core::result::Result<(), ::poot_wgpu_runtime::RuntimeError> {
            // Card 608: the handle comes from `poot_codegen::kernel_handle`
            // against the device pass's own cached `Body`, exactly like `poot-codegen`'s other callers, so
            // this wrapper never constructs a `CompiledKernel` itself and the schema/trap flag can never
            // drift from what the device pass actually compiled. The code is that same `Body` compiled (or
            // served from the kernel cache), sealed to it: raw bytes cannot build a handle (POOT-1207).
            let body = crate::poot_kernel_body(#name_str);
            let artifact = ::poot_codegen::KernelCache::open(::poot_codegen::Target::SpirvVulkan)
                .load_or_compile(
                    &body,
                    ::core::num::NonZeroU64::new(256 << 20).expect("a non-zero artifact limit"),
                )
                .unwrap_or_else(|error| panic!("kernel {:?} did not compile: {error}", #name_str));
            let kernel = ::poot_codegen::kernel_handle(&body, &artifact)
                .expect("the artifact was compiled from this very body");
            ctx.launch(&kernel, &[#(#inputs),*], #output)
        }
    })
}

#[cfg(test)]
mod card537_no_env_input_tests {
    use super::*;

    fn add_fn() -> ItemFn {
        syn::parse_quote! {
            pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
                c[0] = a[0] + b[0];
            }
        }
    }

    /// SC-002: the host-pass wrapper's expansion of a fixture `#[kernel]` function is identical whether or
    /// not `POOT_KERNEL_BUILD` is set in the process environment - `host_wrapper` never reads it. Which
    /// pass a build gets is now this crate's own `device-pass` Cargo feature (see the module doc), a
    /// build-time choice, never a runtime env read; `#[cfg(not(feature = "device-pass"))] pub fn kernel`
    /// (this build) always calls `host_wrapper`, in every process, regardless of environment.
    ///
    /// Mutation (recorded here, never left in the tree): adding an
    /// `if std::env::var_os("POOT_KERNEL_BUILD").is_some() { return Ok(quote!(mod
    /// poot_kernel_build_mutation_marker {})); }` guard back into `host_wrapper` made the two expansions
    /// below differ depending on which call ran with the var set - red (`assert_eq!` failed: left
    /// `"mod poot_kernel_build_mutation_marker { }"`, right `"pub fn add (ctx : & :: poot_wgpu_runtime ::
    /// Context , ..."`); deleting the guard made it green.
    #[test]
    fn host_wrapper_expansion_is_independent_of_poot_kernel_build_env() {
        // SAFETY: test-only env mutation of a process-global; cargo-nextest gives every #[test] its own
        // process (poot-wgpu-device/poot-wgpu-runtime/poot-llm's card 537 tests use the same pattern).
        unsafe {
            std::env::remove_var("POOT_KERNEL_BUILD");
        }
        let without = host_wrapper(&add_fn()).expect("host_wrapper (env unset)");
        unsafe {
            std::env::set_var("POOT_KERNEL_BUILD", "1");
        }
        let with = host_wrapper(&add_fn()).expect("host_wrapper (env set)");
        unsafe {
            std::env::remove_var("POOT_KERNEL_BUILD");
        }
        assert_eq!(with.to_string(), without.to_string());
    }

    /// The device-pass counterpart: `device_fn` (the macro's other half, linked in production only under
    /// `--features device-pass`; compiled here under `cfg(test)` too, see its doc) is likewise a plain
    /// function with no environment read of its own - nothing in either function reads `std::env`, which
    /// is what makes the split-by-feature design in this module's doc comment sound.
    #[test]
    fn device_fn_expansion_is_deterministic() {
        let a = device_fn(add_fn(), &LaunchDecl::default());
        let b = device_fn(add_fn(), &LaunchDecl::default());
        assert_eq!(a.to_string(), b.to_string());
        assert!(a.to_string().contains("__poot_kernel_add"));
    }

    /// The declared launch is recorded as intrinsic calls at the head of the body, in declaration order: the
    /// workgroup shape first, then each LDS array. A kernel with no arguments gets none.
    ///
    /// Mutation: emit the `declare_lds` calls before `declare_workgroup`, or drop the `splice`; the expansion
    /// below no longer starts with the workgroup declaration, or has no declarations at all.
    #[test]
    fn device_fn_records_the_declared_launch_in_order() {
        let decl: LaunchDecl = syn::parse_quote!(workgroup(128, 2), lds(f32, 256), lds(u32, 64));
        let expanded = device_fn(add_fn(), &decl).to_string();
        let workgroup = expanded
            .find("declare_workgroup (128usize , 2usize , 1usize)")
            .expect("the workgroup declaration");
        let first = expanded
            .find("declare_lds :: < f32 > (256usize)")
            .expect("the first array");
        let second = expanded
            .find("declare_lds :: < u32 > (64usize)")
            .expect("the second array");
        let body = expanded.find("c [0] = a [0] + b [0]").expect("the body");
        assert!(
            workgroup < first && first < second && second < body,
            "{expanded}"
        );
        let bare = device_fn(add_fn(), &LaunchDecl::default()).to_string();
        assert!(!bare.contains("declare_"), "{bare}");
    }

    /// A malformed declaration is a compile error naming the problem, not a silently ignored argument.
    #[test]
    fn a_malformed_launch_declaration_is_refused() {
        let parse = |tokens: TokenStream2| {
            syn::parse2::<LaunchDecl>(tokens)
                .err()
                .map(|e| e.to_string())
        };
        assert!(parse(quote!(workgroup(0))).unwrap().contains("at least 1"));
        assert!(
            parse(quote!(workgroup(1, 2, 3, 4)))
                .unwrap()
                .contains("1 to 3")
        );
        assert!(
            parse(quote!(workgroup(64), workgroup(32)))
                .unwrap()
                .contains("twice")
        );
        assert!(parse(quote!(shared(64))).unwrap().contains("unknown"));
        assert!(parse(quote!(lds(f32))).is_some());
        assert!(parse(quote!(workgroup(64), lds(f32, 64))).is_none());
    }
}

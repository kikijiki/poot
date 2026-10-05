use std::fmt::Write as _;

use super::{
    EmitError, Emitter, Local, Operand, Place, ProjectionElem, Rvalue, Statement, Target, Ty,
    align_of, const_ty, fmt_const,
};

impl<'a> Emitter<'a> {
    pub(super) fn emit_len(
        &mut self,
        slice: Local,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        if !self.is_param(slice) {
            return Err(EmitError::Unsupported("Len of a non-param".into()));
        }
        match self.target {
            Target::Nvptx => Ok((format!("%n{}", slice.index), "i64".into())),
            // AIE core: the length is a scalar i32 kernel arg per slice (the IRON harness passes the tile length); same
            // model as NVPTX but 32-bit.
            Target::AieCore => Ok((format!("%n{}", slice.index), "i32".into())),
            // AMDGPU M1: same shape as NVPTX, an i64 kernarg scalar declared alongside the `ptr addrspace(1)` in
            // `emit_signature`.
            Target::AmdGcn(_) => Ok((format!("%n{}", slice.index), "i64".into())),
            Target::SpirvVulkan => {
                let (bufty, mangle) = self.spirv_len_buf();
                let slot = slice.index - 1;
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty} %hlen, i32 {slot})"
                );
                let v = self.fresh();
                let _ = writeln!(w, "  {v} = load i32, ptr addrspace(11) {p}, align 4");
                Ok((v, "i32".into()))
            }
        }
    }

    pub(super) fn operand(
        &mut self,
        op: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        match op {
            Operand::Const(c) => Ok(fmt_const(c, self.usize_llty())),
            Operand::Copy(place) | Operand::Move(place) => self.load_place(place, w),
        }
    }

    pub(super) fn operand_ty(&self, op: &Operand) -> Result<Ty, EmitError> {
        match op {
            Operand::Const(c) => Ok(const_ty(c)),
            Operand::Copy(place) | Operand::Move(place) => self.place_ty(place),
        }
    }

    pub(super) fn place_ty(&self, place: &Place) -> Result<Ty, EmitError> {
        if place.projection.is_empty() {
            Ok(self.body.locals[place.local.index as usize].ty.clone())
        } else if let Ty::Array { elem, .. } = &self.body.locals[place.local.index as usize].ty {
            // [Index] into a private array local -> the element type.
            Ok((**elem).clone())
        } else {
            // [Deref, Index] into a slice param -> the element type.
            self.slice_elem(place.local)
        }
    }

    /// Load the value of a place (scalar local alloca, or a slice element).
    pub(super) fn load_place(
        &mut self,
        place: &Place,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        if place.projection.is_empty() {
            let llty = self.local_llty(place.local)?;
            let v = self.fresh();
            // AMDGPU alloca locals live in addrspace(5) (private); the load's pointer type must match, since the default
            // `ptr` (addrspace 0) is rejected by the verifier.
            if matches!(self.target, Target::AmdGcn(_)) {
                let _ = writeln!(
                    w,
                    "  {v} = load {llty}, ptr addrspace(5) %l{}",
                    place.local.index
                );
            } else {
                let _ = writeln!(w, "  {v} = load {llty}, ptr %l{}", place.local.index);
            }
            Ok((v, llty))
        } else {
            let (ptr, llty, addrspace) = self.element_ptr(place, w)?;
            let v = self.fresh();
            let align = align_of(&llty);
            let _ = writeln!(
                w,
                "  {v} = load {llty}, ptr addrspace({addrspace}) {ptr}, align {align}"
            );
            Ok((v, llty))
        }
    }

    /// Store a value into a place.
    pub(super) fn store_place(
        &mut self,
        place: &Place,
        val: &str,
        llty: &str,
        w: &mut String,
    ) -> Result<(), EmitError> {
        if place.projection.is_empty() {
            if matches!(self.target, Target::AmdGcn(_)) {
                let _ = writeln!(
                    w,
                    "  store {llty} {val}, ptr addrspace(5) %l{}",
                    place.local.index
                );
            } else {
                let _ = writeln!(w, "  store {llty} {val}, ptr %l{}", place.local.index);
            }
            Ok(())
        } else {
            let (ptr, elem_llty, addrspace) = self.element_ptr(place, w)?;
            let align = align_of(&elem_llty);
            let _ = writeln!(
                w,
                "  store {elem_llty} {val}, ptr addrspace({addrspace}) {ptr}, align {align}"
            );
            Ok(())
        }
    }

    /// Compute the pointer to a slice element `(*param)[idx]`, returning (ptr_reg, elem_llty, addrspace).
    pub(super) fn element_ptr(
        &mut self,
        place: &Place,
        w: &mut String,
    ) -> Result<(String, String, u32), EmitError> {
        // A private array local indexed as `arr[idx]` (projection [Index(idx)], no Deref): GEP into the alloca.
        // NVPTX-only (the SpirvVulkan rejection is in emit_entry's alloca loop).
        if let Ty::Array { elem, len } = self.body.locals[place.local.index as usize].ty.clone() {
            let idx_local = match place.projection.as_slice() {
                [ProjectionElem::Index(i)] => *i,
                other => {
                    return Err(EmitError::Unsupported(format!(
                        "array projection {other:?}"
                    )));
                }
            };
            let elem_llty = self.scalar_llty(&elem)?;
            let (idx_val, _) = self.load_place(&Place::local(idx_local), w)?;
            let us = self.usize_llty();
            let p = self.fresh();
            // AMDGPU: GEP base pointer must carry addrspace(5) to match the alloca's addrspace(5).
            if matches!(self.target, Target::AmdGcn(_)) {
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr inbounds [{len} x {elem_llty}], ptr addrspace(5) %l{}, {us} 0, {us} {idx_val}",
                    place.local.index
                );
            } else {
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr inbounds [{len} x {elem_llty}], ptr %l{}, {us} 0, {us} {idx_val}",
                    place.local.index
                );
            }
            return Ok((
                p,
                elem_llty,
                if matches!(self.target, Target::AmdGcn(_)) {
                    5
                } else {
                    0
                },
            ));
        }
        // projection must be [Deref, Index(idx)].
        let idx_local = match place.projection.as_slice() {
            [ProjectionElem::Deref, ProjectionElem::Index(i)] => *i,
            other => return Err(EmitError::Unsupported(format!("projection {other:?}"))),
        };
        let elem = self.slice_elem(place.local)?;
        let elem_llty = self.scalar_llty(&elem)?;
        // load the index value (usize-width).
        let (idx_val, _) = self.load_place(&Place::local(idx_local), w)?;
        match self.target {
            Target::Nvptx => {
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr inbounds {elem_llty}, ptr addrspace(1) %p{}, i64 {idx_val}",
                    place.local.index
                );
                Ok((p, elem_llty, 1))
            }
            // AIE core: flat `ptr` (default addrspace 0), i32 index (32-bit core).
            Target::AieCore => {
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr inbounds {elem_llty}, ptr %p{}, i32 {idx_val}",
                    place.local.index
                );
                Ok((p, elem_llty, 0))
            }
            // AMDGPU M1: global memory lives in addrspace(1), as on NVPTX and as the AMDGPU backend defaults for
            // `amdgcn-amd-amdhsa`. i64 indexing matches the `usize` lower bound (see `scalar_llty`). Element ptrs are
            // `ptr addrspace(1)`, loaded with `align 4` like NVPTX.
            Target::AmdGcn(_) => {
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr inbounds {elem_llty}, ptr addrspace(1) %p{}, i64 {idx_val}",
                    place.local.index
                );
                Ok((p, elem_llty, 1))
            }
            Target::SpirvVulkan => {
                let (bufty, mangle) = self.spirv_buf(&elem, true)?;
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty} %h{}, i32 {idx_val})",
                    place.local.index
                );
                Ok((p, elem_llty, 11))
            }
        }
    }

    /// Address a `Ty::Vec` access `(*place.local)[i]` at vector-group index `i` (spec 134 P1,
    /// `Rvalue::VectorLoad`/`Statement::VectorStore`): elements `[i*lanes, i*lanes+lanes)` of the underlying scalar
    /// buffer, reinterpreted as one `vec_ty` value. SpirvVulkan-only for now (FR-007). The resource for this access
    /// is declared with the `<lanes x elem>` element type (FR-004, the A1 probe fixture), which may differ from the
    /// scalar-typed resource `element_ptr`/`param_bind_ty` would pick. A body must not mix scalar and vector access
    /// to one param yet (FR-004b's two-binding fallback is not wired; `param_bind_ty` picks one binding type per
    /// param).
    pub(super) fn vector_element_ptr(
        &mut self,
        place: &Place,
        vec_ty: &Ty,
        w: &mut String,
    ) -> Result<(String, String, u32), EmitError> {
        if self.target != Target::SpirvVulkan {
            return Err(EmitError::Unsupported(format!(
                "Ty::Vec buffer access on {:?}: vector codegen is SpirvVulkan-only in this increment \
                 (spec 134 P1); NVPTX/AMDGCN is a later phase",
                self.target
            )));
        }
        let idx_local = match place.projection.as_slice() {
            [ProjectionElem::Deref, ProjectionElem::Index(i)] => *i,
            other => {
                return Err(EmitError::Unsupported(format!(
                    "vector projection {other:?}"
                )));
            }
        };
        let elem_llty = self.scalar_llty(vec_ty)?;
        let (idx_val, _) = self.load_place(&Place::local(idx_local), w)?;
        let (bufty, mangle) = self.spirv_buf(vec_ty, true)?;
        let p = self.fresh();
        let _ = writeln!(
            w,
            "  {p} = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty} %h{}, i32 {idx_val})",
            place.local.index
        );
        Ok((p, elem_llty, 11))
    }

    /// The `Ty` used to declare param `p`'s SPIR-V resource binding: the scalar slice element type (`slice_elem`),
    /// unless the body accesses `p` via `VectorLoad`/`VectorStore`, in which case that access's vector `Ty`
    /// (elem+lanes) is used (FR-004: the resource must be declared with the vector element type, not bitcast from a
    /// scalar one). Only the first vector access found is consulted; mixing scalar and vector access on one param
    /// needs FR-004b's two-binding mechanism, deferred to the P1 dequant-kernel adoption.
    pub(super) fn param_bind_ty(&self, p: Local) -> Result<Ty, EmitError> {
        for bb in &self.body.blocks {
            for s in &bb.statements {
                match s {
                    Statement::Assign(dest, Rvalue::VectorLoad { place }) if place.local == p => {
                        return Ok(self.body.locals[dest.local.index as usize].ty.clone());
                    }
                    Statement::VectorStore { place, value } if place.local == p => {
                        return self.operand_ty(value);
                    }
                    _ => {}
                }
            }
        }
        self.slice_elem(p)
    }
}

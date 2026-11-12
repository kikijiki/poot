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
            // A narrowed body truncates the ABI's i64 length to its i32 `usize`: the planner proved it fits.
            Target::Nvptx if self.narrowed() => {
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = trunc i64 %n{} to i32", slice.index);
                Ok((t, "i32".into()))
            }
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
                let us = self.usize_llty();
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr inbounds {elem_llty}, ptr addrspace(1) %p{}, {us} {idx_val}",
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

    /// Address a `Ty::Vec` store `(*place.local)[i]` at vector-group index `i` (spec 134 P1,
    /// `Statement::VectorStore`): elements `[i*lanes, i*lanes+lanes)` of the underlying scalar buffer, reinterpreted
    /// as one `vec_ty` value. On NVPTX it is a GEP over `<lanes x elem>` into the param's own pointer. On SpirvVulkan
    /// the resource for this access is declared with the `<lanes x elem>` element type (FR-004, the A1 probe fixture),
    /// which differs from the scalar-typed resource `element_ptr`/`param_bind_ty` would pick, so a body must not mix
    /// scalar and vector access to one param there (FR-004b's two-binding fallback is not wired). Other targets
    /// refuse `Ty::Vec` in `scalar_llty`.
    pub(super) fn vector_element_ptr(
        &mut self,
        place: &Place,
        vec_ty: &Ty,
        w: &mut String,
    ) -> Result<(String, String, u32), EmitError> {
        let idx_local = vector_index_local(place)?;
        // `scalar_llty` refuses `Ty::Vec` on a target without vector codegen.
        let elem_llty = self.scalar_llty(vec_ty)?;
        let (idx_val, idx_llty) = self.load_place(&Place::local(idx_local), w)?;
        if self.target == Target::Nvptx {
            // The group index steps over `<lanes x elem>`: group `i` starts at element `i * lanes` of the param's
            // `ptr addrspace(1)`, which the same param's scalar reads also index (NVPTX has no typed resource).
            let p = self.fresh();
            let _ = writeln!(
                w,
                "  {p} = getelementptr inbounds {elem_llty}, ptr addrspace(1) %p{}, {idx_llty} {idx_val}",
                place.local.index
            );
            return Ok((p, elem_llty, 1));
        }
        self.spirv_vector_group_ptr(place.local, vec_ty, &idx_val, w)
    }

    /// The pointer to group `group` of the vector resource of param `param` (declared by `param_bind_ty`).
    fn spirv_vector_group_ptr(
        &mut self,
        param: Local,
        vec_ty: &Ty,
        group: &str,
        w: &mut String,
    ) -> Result<(String, String, u32), EmitError> {
        let elem_llty = self.scalar_llty(vec_ty)?;
        let (bufty, mangle) = self.spirv_buf(vec_ty, true)?;
        let p = self.fresh();
        let _ = writeln!(
            w,
            "  {p} = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty} %h{}, i32 {group})",
            param.index
        );
        Ok((p, elem_llty, 11))
    }

    /// Lower [`poot_kernel_ir::Rvalue::VectorLoad`] of `dest_ty` (`lanes` of `elem`) at the element index `place`
    /// holds, whose address the emitter proved `align_bytes`-aligned.
    ///
    /// NVPTX and AMDGPU load `<lanes x elem>` through the same element pointer a scalar read of the param uses, so
    /// one param may be read both ways ([`crate::Capability::mixed_vector_reads`]); the backend emits one vector
    /// instruction where `align_bytes` covers the whole vector and scalar loads where it does not. SpirvVulkan has
    /// typed resources: a param every access of which is a vector-aligned load of one vector type is declared with
    /// that vector element and read by group; any other param reads its lanes as scalar loads (same values, no
    /// vector instruction). Other targets refuse `Ty::Vec` in `scalar_llty`.
    pub(super) fn emit_vector_load(
        &mut self,
        place: &Place,
        align_bytes: u32,
        dest_ty: &Ty,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        let Ty::Vec { elem, lanes } = dest_ty else {
            return Err(EmitError::Unsupported(
                "VectorLoad requires a Ty::Vec destination local".into(),
            ));
        };
        let vec_llty = self.scalar_llty(dest_ty)?;
        let v = self.fresh();
        if self.target != Target::SpirvVulkan {
            let (ptr, _, addrspace) = self.element_ptr(place, w)?;
            let _ = writeln!(
                w,
                "  {v} = load {vec_llty}, ptr addrspace({addrspace}) {ptr}, align {align_bytes}"
            );
            return Ok((v, vec_llty));
        }
        let idx_local = vector_index_local(place)?;
        let (idx_val, _) = self.load_place(&Place::local(idx_local), w)?;
        if self.vector_binding(place.local).is_some() {
            // The binding proof gave every load of this param an alignment of a whole vector, so the element index
            // is a multiple of `lanes` and the group index is its quotient.
            let group = self.fresh();
            let _ = writeln!(
                w,
                "  {group} = lshr i32 {idx_val}, {}",
                lanes.trailing_zeros()
            );
            let (ptr, _, addrspace) =
                self.spirv_vector_group_ptr(place.local, dest_ty, &group, w)?;
            let _ = writeln!(
                w,
                "  {v} = load {vec_llty}, ptr addrspace({addrspace}) {ptr}, align {align_bytes}"
            );
            return Ok((v, vec_llty));
        }
        let lane_llty = self.scalar_llty(elem)?;
        let (bufty, mangle) = self.spirv_buf(elem, true)?;
        let lane_align = align_of(&lane_llty);
        let mut vector = "poison".to_string();
        for lane in 0..*lanes {
            let at = self.fresh();
            let _ = writeln!(w, "  {at} = add i32 {idx_val}, {lane}");
            let ptr = self.fresh();
            let _ = writeln!(
                w,
                "  {ptr} = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty} %h{}, i32 {at})",
                place.local.index
            );
            let value = self.fresh();
            let _ = writeln!(
                w,
                "  {value} = load {lane_llty}, ptr addrspace(11) {ptr}, align {lane_align}"
            );
            let next = self.fresh();
            let _ = writeln!(
                w,
                "  {next} = insertelement {vec_llty} {vector}, {lane_llty} {value}, i32 {lane}"
            );
            vector = next;
        }
        Ok((vector, vec_llty))
    }

    /// The vector type param `p`'s SpirvVulkan resource is declared with, when every access of `p` is a load or
    /// store of that one vector type and every load proves an alignment of a whole vector (so its element index
    /// divides into a group index). `None` for a param with any other access: it keeps its scalar resource.
    fn vector_binding(&self, p: Local) -> Option<Ty> {
        let mut found: Option<Ty> = None;
        let mut vector_accesses = 0usize;
        let mut accesses = 0usize;
        for statement in self.body.blocks.iter().flat_map(|b| &b.statements) {
            statement.for_each_read(|l| accesses += usize::from(l == p));
            let (place, ty) = match statement {
                Statement::Assign(dest, Rvalue::VectorLoad { place, align_bytes }) => {
                    let ty = self.body.locals[dest.local.index as usize].ty.clone();
                    let Ty::Vec { elem, lanes } = &ty else {
                        return None;
                    };
                    let whole = u64::from(*lanes) * u64::from(elem.buffer_element_bytes()?);
                    if u64::from(*align_bytes) < whole {
                        return None;
                    }
                    (place, ty)
                }
                Statement::VectorStore { place, value } => (place, self.operand_ty(value).ok()?),
                _ => continue,
            };
            if place.local != p {
                continue;
            }
            if found.get_or_insert_with(|| ty.clone()) != &ty {
                return None;
            }
            vector_accesses += 1;
        }
        found.filter(|_| accesses == vector_accesses)
    }

    /// The `Ty` used to declare param `p`'s SPIR-V resource binding: its [`Self::vector_binding`] when it has one
    /// (FR-004: the resource must be declared with the vector element type, not bitcast from a scalar one), else
    /// the scalar slice element type.
    pub(super) fn param_bind_ty(&self, p: Local) -> Result<Ty, EmitError> {
        match self.vector_binding(p) {
            Some(ty) => Ok(ty),
            None => self.slice_elem(p),
        }
    }
}

/// The local holding the index of a vector access `place`, which must project `[Deref, Index(i)]`.
fn vector_index_local(place: &Place) -> Result<Local, EmitError> {
    match place.projection.as_slice() {
        [ProjectionElem::Deref, ProjectionElem::Index(i)] => Ok(*i),
        other => Err(EmitError::Unsupported(format!(
            "vector projection {other:?}"
        ))),
    }
}

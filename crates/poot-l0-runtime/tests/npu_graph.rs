//! Intel-NPU rows for `ZE_extension_graph`: an OpenVINO graph the VPU emitter wrote, compiled by the driver's
//! graph compiler and run on the NPU through `poot-l0-runtime`, against the values the graph computes.
//!
//! `#[ignore]`d: they need the Intel NPU, the Level Zero loader, the NPU driver and its compiler on the library
//! path, none of which the model-free lane has. Run them on the Lunar Lake laptop with `just test-device-l0`.

use std::time::Duration;

use poot_codegen::{VpuElem, VpuInput, VpuNode, VpuOp, VpuRegion, VpuTensor, VpuValue, emit_vpu};
use poot_l0_runtime::{GraphDirection, GraphIr, GraphPrecision, L0Context, L0Error};
use poot_quant::scalar::f32_to_f16;

const N: usize = 16;

/// The IR of `a + b` over two F16 `[1, 16]` inputs.
fn add_ir() -> poot_codegen::VpuIr {
    let tensor = VpuTensor {
        elem: VpuElem::F16,
        shape: vec![1, N],
    };
    let region = VpuRegion {
        name: "add".into(),
        inputs: ["a", "b"]
            .into_iter()
            .map(|name| VpuInput {
                name: name.into(),
                tensor: tensor.clone(),
            })
            .collect(),
        nodes: vec![VpuNode {
            origin: 2,
            op: VpuOp::Add,
            inputs: vec![VpuValue(0), VpuValue(1)],
            out: tensor,
        }],
        outputs: vec![VpuValue(2)],
    };
    emit_vpu(&region).expect("the add region is well formed")
}

fn words(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(|v| f32_to_f16(v).to_le_bytes()).collect()
}

fn npu() -> L0Context {
    L0Context::open().expect("an NPU context")
}

/// An IR graph compiles, reports the arguments the compiler built, runs, and its native blob loads without the
/// compiler to the same answer.
#[test]
#[ignore = "needs the Intel NPU with its Level Zero driver and graph compiler (the Lunar Lake laptop)"]
fn an_ir_graph_compiles_runs_and_its_blob_reloads_to_the_same_answer() {
    let context = npu();
    assert_eq!(context.vendor_id(), 0x8086);
    let properties = context.graph_device_properties().unwrap();
    assert!(properties.compiler_version.0 > 0, "{properties:?}");

    let ir = add_ir();
    let ir = GraphIr {
        xml: ir.xml.as_bytes(),
        weights: &ir.weights,
    };
    let compiled = context.compile_graph(&ir, "").expect("the add compiles");
    let arguments: Vec<_> = compiled
        .arguments()
        .iter()
        .map(|a| {
            (
                a.name.as_str(),
                a.direction,
                a.precision,
                a.dims[..2].to_vec(),
            )
        })
        .collect();
    assert_eq!(
        arguments,
        [
            (
                "a",
                GraphDirection::Input,
                GraphPrecision::Fp16,
                vec![1, 16]
            ),
            (
                "b",
                GraphDirection::Input,
                GraphPrecision::Fp16,
                vec![1, 16]
            ),
            (
                "v2",
                GraphDirection::Output,
                GraphPrecision::Fp16,
                vec![1, 16]
            ),
        ]
    );

    let run = |graph: &poot_l0_runtime::L0Graph| {
        let mut buffers: Vec<_> = (0..3).map(|_| context.alloc_host(2 * N).unwrap()).collect();
        let a = words((0..N).map(|i| i as f32));
        let b = words((0..N).map(|i| 0.5 * i as f32));
        buffers[0]
            .write(0..2 * N, |bytes| bytes.copy_from_slice(&a))
            .unwrap();
        buffers[1]
            .write(0..2 * N, |bytes| bytes.copy_from_slice(&b))
            .unwrap();
        let mut recording = context.begin_recording().unwrap();
        recording
            .execute_graph(graph, &[&buffers[0], &buffers[1], &buffers[2]])
            .unwrap();
        let recording = recording.finish().unwrap();
        context
            .execute(&recording, Duration::from_secs(10))
            .unwrap();
        let mut out = vec![0u8; 2 * N];
        buffers[2].read(0, &mut out).unwrap();
        out
    };
    let want = words((0..N).map(|i| 1.5 * i as f32));
    assert_eq!(run(&compiled), want);

    let blob = compiled.native_binary().unwrap();
    assert!(blob.len() > 1000, "a whole blob: {} bytes", blob.len());
    // SAFETY: `blob` is the unmodified `native_binary` of a graph this context's compiler just built.
    let loaded = unsafe { context.load_native_graph(&blob) }.expect("the blob loads");
    assert_eq!(loaded.arguments(), compiled.arguments());
    assert_eq!(
        run(&loaded),
        want,
        "the reloaded blob computes the same answer"
    );
}

/// A recording binds its buffers when an execution is appended and replays over whatever those buffers hold
/// now; the runtime refuses what belongs to another context, a wrong argument count and a mismatched copy before
/// the driver sees them.
#[test]
#[ignore = "needs the Intel NPU with its Level Zero driver and graph compiler (the Lunar Lake laptop)"]
fn a_recording_replays_over_new_inputs_and_refuses_what_is_not_its_own() {
    let context = npu();
    let ir = add_ir();
    let graph = context
        .compile_graph(
            &GraphIr {
                xml: ir.xml.as_bytes(),
                weights: &ir.weights,
            },
            "",
        )
        .unwrap();
    let mut a = context.alloc_host(2 * N).unwrap();
    let mut b = context.alloc_host(2 * N).unwrap();
    let out = context.alloc_host(2 * N).unwrap();

    let mut builder = context.begin_recording().unwrap();
    assert!(matches!(
        builder.execute_graph(&graph, &[&a, &b]),
        Err(L0Error::ArgumentCount {
            expected: 3,
            given: 2
        })
    ));
    let other = npu();
    let foreign = other.alloc_host(2 * N).unwrap();
    assert!(matches!(
        builder.execute_graph(&graph, &[&a, &foreign, &out]),
        Err(L0Error::ForeignResource { what: "buffer" })
    ));
    let short = context.alloc_host(2 * N - 2).unwrap();
    assert!(matches!(
        builder.copy(&short, &a),
        Err(L0Error::TransferLength { .. })
    ));
    builder.execute_graph(&graph, &[&a, &b, &out]).unwrap();
    let recording = builder.finish().unwrap();
    assert!(matches!(
        other.execute(&recording, Duration::from_secs(1)),
        Err(L0Error::ForeignResource { what: "recording" })
    ));

    for round in 0..3 {
        let base = 10.0 * round as f32;
        a.write(0..2 * N, |bytes| {
            bytes.copy_from_slice(&words((0..N).map(|i| base + i as f32)))
        })
        .unwrap();
        b.write(0..2 * N, |bytes| {
            bytes.copy_from_slice(&words((0..N).map(|_| 1.0)))
        })
        .unwrap();
        context
            .execute(&recording, Duration::from_secs(10))
            .unwrap();
        let mut got = vec![0u8; 2 * N];
        out.read(0, &mut got).unwrap();
        assert_eq!(
            got,
            words((0..N).map(|i| base + i as f32 + 1.0)),
            "replay {round}"
        );
    }
}

/// The compiler lists the layers it takes, and a graph it cannot parse fails with its own log naming the layer.
#[test]
#[ignore = "needs the Intel NPU with its Level Zero driver and graph compiler (the Lunar Lake laptop)"]
fn the_compiler_lists_its_layers_and_names_what_it_cannot_parse() {
    let context = npu();
    let ir = add_ir();
    let supported = context
        .query_supported_layers(
            &GraphIr {
                xml: ir.xml.as_bytes(),
                weights: &ir.weights,
            },
            "",
        )
        .unwrap();
    assert!(supported.iter().any(|name| name == "v2"), "{supported:?}");

    let bad = ir.xml.replace("type=\"Add\"", "type=\"Bucketize\"");
    let Err(L0Error::GraphBuild { log, .. }) = context.compile_graph(
        &GraphIr {
            xml: bad.as_bytes(),
            weights: &ir.weights,
        },
        "",
    ) else {
        panic!("a layer type outside the opset must fail to build");
    };
    assert!(log.contains("Bucketize"), "{log}");
}

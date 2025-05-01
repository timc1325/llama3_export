import ezkl
import asyncio

# Define paths
paths = {
    "onnx": "network.onnx",
    "settings": "settings.json",
    "compiled": "network.compiled",
    "input": "input.json",
    "witness": "witness.json",
    "pk": "pk.key",
    "vk": "vk.key",
    "proof": "proof.json"
}

def generate_settings():
    print("Generating settings...")
    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "private"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"
    run_args.num_inner_cols = 2
    run_args.variables = [("batch_size", 2)]

    if not ezkl.gen_settings(
        model=paths["onnx"],
        output=paths["settings"],
        py_run_args=run_args
    ):
        raise RuntimeError("Failed to generate settings.")

async def calibrate_settings():
    print("Calibrating settings...")
    result = await ezkl.calibrate_settings(
        data=paths["input"],
        model=paths["onnx"],
        settings=paths["settings"],
        target="resources"
    )
    if not result:
        raise RuntimeError("Calibration failed.")

def compile_circuit():
    print("Compiling circuit...")
    if not ezkl.compile_circuit(
        model=paths["onnx"],
        compiled_circuit=paths["compiled"],
        settings_path=paths["settings"]
    ):
        raise RuntimeError("Failed to compile circuit.")

async def generate_witness():
    print("Generating witness...")
    await ezkl.gen_witness(
        data=paths["input"],
        model=paths["compiled"],
        output=paths["witness"]
    )

def mock_check():
    print("Running mock check...")
    res = ezkl.mock(
        witness=paths["witness"],
        model=paths["compiled"]
    )
    if not res:
        raise RuntimeError("Mock prover failed.")
    print("Mock prover passed ✅.")

def setup_circuit():
    print("Setting up circuit...")
    ezkl.setup(
        model=paths["compiled"],
        vk_path=paths["vk"],
        pk_path=paths["pk"]
    )

def prove_circuit():
    print("Proving circuit...")
    ezkl.prove(
        witness_path=paths["witness"],
        compiled_model_path=paths["compiled"],
        pk_path=paths["pk"],
        proof_path=paths["proof"],
        strategy="single"
    )

async def main():
    generate_settings()
    await calibrate_settings()
    compile_circuit()
    await ezkl.get_srs(paths["settings"])  # Ensures SRS is ready
    await generate_witness()
    mock_check()
    setup_circuit()
    prove_circuit()
    print("✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())

import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- CONFIGURATION ---
MODEL_FP32  = 'output/vae_model.onnx'
MODEL_QUANT = 'output/vae_model_quant.onnx'


def run_quantization():
    if not os.path.exists(MODEL_FP32):
        print(f"ERROR: {MODEL_FP32} not found — run 03_optimised_vae.py first.")
        return

    print(f"Quantising {MODEL_FP32} to UINT8...")
    quantize_dynamic(MODEL_FP32, MODEL_QUANT, weight_type=QuantType.QUInt8)

    size_fp32  = os.path.getsize(MODEL_FP32)  / 1024
    size_quant = os.path.getsize(MODEL_QUANT) / 1024
    reduction  = (1 - size_quant / size_fp32) * 100

    print(f"\n{'Original (FP32)':<25} {size_fp32:.2f} KB")
    print(f"{'Quantised (UINT8)':<25} {size_quant:.2f} KB")
    print(f"{'Size reduction':<25} {reduction:.1f}%")
    print(f"\nQuantised model saved: {MODEL_QUANT}")


if __name__ == "__main__":
    run_quantization()
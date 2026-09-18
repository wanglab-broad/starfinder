"""Private benchmark report formatting."""
import numpy as np

def _print_quality_report(report: dict[str, dict[str, float]]) -> None:
    """Print a formatted registration quality report.

    Parameters
    ----------
    report : dict
        Output from registration_quality_report().
    """
    print("=" * 60)
    print("REGISTRATION QUALITY REPORT")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
    print("-" * 60)

    # NCC
    b, a = report["ncc"]["before"], report["ncc"]["after"]
    print(f"{'NCC (↑ better)':<25} {b:>12.4f} {a:>12.4f} {(a-b)/abs(b+1e-10)*100:>+11.1f}%")

    # SSIM
    b, a = report["ssim"]["before"], report["ssim"]["after"]
    print(f"{'SSIM (↑ better)':<25} {b:>12.4f} {a:>12.4f} {(a-b)/max(abs(b),0.001)*100:>+11.1f}%")

    # Spot IoU
    b, a = report["spot_iou"]["before"], report["spot_iou"]["after"]
    print(f"{'Spot IoU (↑ better)':<25} {b:>12.4f} {a:>12.4f} {(a-b)/max(b,0.001)*100:>+11.1f}%")

    # Spot Dice
    b, a = report["spot_dice"]["before"], report["spot_dice"]["after"]
    print(f"{'Spot Dice (↑ better)':<25} {b:>12.4f} {a:>12.4f} {(a-b)/max(b,0.001)*100:>+11.1f}%")

    # Match rate
    b, a = report["match_rate"]["before"], report["match_rate"]["after"]
    print(f"{'Match Rate (↑ better)':<25} {b*100:>11.1f}% {a*100:>11.1f}% {(a-b)/max(b,0.001)*100:>+11.1f}%")

    # Match distance
    b, a = report["match_distance"]["before"], report["match_distance"]["after"]
    b_str = f"{b:.2f}" if not np.isnan(b) else "N/A"
    a_str = f"{a:.2f}" if not np.isnan(a) else "N/A"
    print(f"{'Match Distance (↓ better)':<25} {b_str:>12} {a_str:>12}")

    print("-" * 60)

    # Barcode decoding projection
    b_rate = report["match_rate"]["before"]
    a_rate = report["match_rate"]["after"]
    print("\n📈 Projected barcode decoding (4 rounds):")
    print(f"   Before: {b_rate*100:.1f}%/round → {b_rate**4*100:.1f}% decoded")
    print(f"   After:  {a_rate*100:.1f}%/round → {a_rate**4*100:.1f}% decoded")

    print("=" * 60)

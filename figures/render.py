"""Draw the final figures from their author-supplied input tables."""
import argparse
from pathlib import Path
import pandas as pd
import figures_3_4 as shared
from matplotlib import pyplot as plt

def attribution(archive):
    pi = archive.query("panel == '5b'").rename(columns={"predictor":"variable", "estimate":"PI_mean","SD":"PI_SD"})
    mdi = archive.query("panel == '5c'").rename(columns={"predictor":"variable", "estimate":"MDI"})
    keys = ["habitat", "variable"]
    return pi[keys+["PI_mean","PI_SD"]].merge(mdi[keys+["MDI"]], on=keys, validate="1:1")


def figure5(book, output):
    from figure5 import ORDER, render
    data = attribution(book.parse("模型归因"))
    data["label"] = data.variable.map(dict(ORDER))
    render(book.parse("图5汇总"), data, output)


def figure6(book, output):
    from figure6 import draw
    perf = book.parse("模型测试指标").query("figure == 'Figure 6'").rename(
        columns={"test_n":"n", "test_R2":"R2", "test_RMSE":"RMSE", "test_MAE":"MAE"})
    fields = ["observation_id", "habitat", "role_102", "Z_fit"]
    scatter = book.parse("建模逐条")[fields].merge(
        book.parse("预测逐条")[["observation_id", "cascade_Z_102"]], on="observation_id", validate="1:1")
    scatter = scatter.rename(columns={"role_102":"role", "Z_fit":"Z", "cascade_Z_102":"predicted"}).assign(seed=102)
    source = book.parse("模型归因")
    shap = source[source.panel.isin(["6ef", "6g"])].rename(columns={"estimate":"group_mean_absolute_shapley"})
    perm = source.query("panel == '6h'").rename(
        columns={"output":"component", "predictor":"category", "estimate":"delta_r2", "SD":"repeat_sd"})
    draw(perf, scatter, shap, perm, output)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", choices=("1", "2", "3", "4", "5", "6", "7", "S11", "S12", "S13"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, help="Analysis workbook for Figures 5 and 6")
    parser.add_argument("--tables", type=Path, help="Directory with the CSV inputs for Figures 3, 4 and 7")
    parser.add_argument("--base", type=Path, help="Figure 2 base plate stem (PNG and SVG)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.number in ("5", "6"):
        if args.workbook is None:
            parser.error("--workbook is required for Figures 5 and 6")
        with pd.ExcelFile(args.workbook) as book:
            (figure5 if args.number == "5" else figure6)(book, args.output)
        return
    if args.number == "1":
        from figure1 import render
        render(args.output)
        return
    if args.tables is None:
        parser.error("--tables is required for Figures 3, 4 and 7")
    def read(name):
        return pd.read_csv(args.tables / (name + ".csv"))
    def finish(fig, name):
        for ext in ("svg", "png", "pdf"):
            fig.savefig(args.output / f"{name}.{ext}", dpi=260, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    if args.number == "2":
        if args.base is None:
            parser.error("--base is required for Figure 2")
        from figure2 import render
        render(args.base, args.tables, args.output)
        return
    with plt.rc_context(shared.STYLE):
        if args.number == "3":
            from figure3 import render
            render(read("primary_scores"), read("matched_pairs"), read("paired_rms").set_index("contrast"), finish)
        elif args.number == "4":
            shared.save = lambda fig, stem, dpi=300: finish(fig, "Figure_4")
            stress = read("weight_stress").query("tier == 'Tier 2'").copy()
            stress["scenario"] = stress["scenario"].replace({
                "Worst Case": "Maximum", "Best Case": "Minimum",
                "Increased Interactions": "Sharper gate",
            })
            shared.figure4(read("sobol"), stress,
                           "Combined", "WaterSediment", (-.2, 1.15), (0, 100))
        elif args.number == "S11":
            from figure_s11 import render
            render(read("external_exposures"), args.output)
        elif args.number == "S13":
            from figure_s13 import render
            render(read("regional_quality"), args.output)
        elif args.number == "S12":
            from figure_s12 import render
            render(read("dml_coefficients"), read("spatial_diagnostics"),
                   read("province_sensitivity"), read("spatial_intervals"), args.output)
        else:
            from figure7 import render
            render(read("dml_coefficients"), read("regional_scores"), args.output)


if __name__ == "__main__":
    main()

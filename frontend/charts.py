#!/user/bin/env python3

from collections import defaultdict

import altair
import pandas as pd

import parse_csv


def make_scatterplot_base(data, color_key):
    altair.renderers.enable("html")
    value_counts = defaultdict(lambda: 0)
    items = []
    for x in data:
        answer = x["rec"].get(color_key, "Unknown")
        answer_values = parse_csv.split_values(answer)
        for v in answer_values:
            value_counts[v] += 1
        items.extend(
            [
                {
                    "ColorGroup": group_value,
                    "Groups": answer,
                    "x": x["vec"][0],
                    "y": x["vec"][1],
                    "Response": x["sentence"],
                }
                for group_value in answer_values
            ]
        )

    # Display the value count in the legend
    legend_values = {}
    for item in items:
        legend_value = "%s [%0.2f%%]" % (
            item["ColorGroup"],
            100.0 * value_counts[item["ColorGroup"]] / len(data),
        )
        legend_values[legend_value] = value_counts[item["ColorGroup"]]
        item["GroupForLegend"] = legend_value
    df = pd.DataFrame(items)

    # Sort legend values by frequency (descending) so that most frequent are
    # on the top of the legend.
    sorted_values = list(legend_values.keys())
    sorted_values.sort(key=lambda x: legend_values[x], reverse=True)

    selection = altair.selection_multi(fields=["GroupForLegend"])
    coloring_scheme = altair.Color(
        "GroupForLegend",
        type="nominal",
        scale=altair.Scale(scheme="tableau20"),
        sort=sorted_values,
    )
    color = altair.condition(selection, coloring_scheme, altair.value("white"))
    # TODO.  Ensure "unclustered" is grey.  Below doesn't work.
    # color = altair.condition(altair.datum.GroupForLegend == '-1', altair.value('grey'), color)

    chart = (
        (
            altair.Chart(df, height=400, width=1200)
            .mark_circle(size=80, opacity=1.0)
            .encode(
                x=altair.X("x", axis=None, scale=altair.Scale(zero=False)),
                y=altair.Y("y", axis=None, scale=altair.Scale(zero=False)),
                color=color,
                tooltip=["Response", "Groups"],
            )
            .interactive()
            .configure_view(strokeOpacity=1)
        )
        .configure_legend(
            orient="right",
            title=None,
            labelLimit=300,
            cornerRadius=10,
        )
        .add_selection(selection)
    )
    return chart

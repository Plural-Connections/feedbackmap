#!/user/bin/env python3

from collections import defaultdict
import re

import altair
import pandas as pd

import parse_csv

# It shouldn't be necessary to enumerate the tableau20 colors like tis
# but I can't figure out how to get them programmatically
_TABLEAU20_SCHEME = ["#4379AB", "#96CCEB", "#FF8900", "#FFBC71", "#3DA443", "#76D472", "#BA9900", "#F7CD4B", "#249A95", "#77BEB6", "#F14A54", "#FF9797", "#7B706E", "#BCB0AB", "#E16A96", "#FFBCD3", "#B976A3", "#DCA3CA", "#A3745C", "#DDB3A4"]

def make_scatterplot_base(data, color_key):
    altair.renderers.enable("html")
    value_counts = defaultdict(lambda: 0)
    items = []
    for x in data:
        categories = x["rec"].get(color_key, "Unknown")
        categories_values = parse_csv.split_values(categories)
        for v in categories_values:
            value_counts[v] += 1
        items.extend(
            [
                {
                    "ColorGroup": group_value,
                    "Categories": categories.replace(";", "; "),
                    "x": x["vec"][0],
                    "y": x["vec"][1],
                    "Answer": x["sentence"],
                }
                for group_value in categories_values
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
                tooltip=["Answer", "Categories"],
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

    category_to_color = dict([(re.sub(r" \[.*", "", category),
                               _TABLEAU20_SCHEME[i])
                              for i, category in enumerate(sorted_values[:len(_TABLEAU20_SCHEME)])])

    return chart, category_to_color

#!/user/bin/env python3

from collections import defaultdict
import re

import altair
import pandas as pd

import app_config
import parse_csv
import streamlit as st

# It shouldn't be necessary to enumerate the tableau20 colors like tis
# but I can't figure out how to get them programmatically
_TABLEAU20_SCHEME = [
    "#4379AB",
    "#96CCEB",
    "#FF8900",
    "#FFBC71",
    "#3DA443",
    "#76D472",
    "#BA9900",
    "#F7CD4B",
    "#249A95",
    "#77BEB6",
    "#F14A54",
    "#FF9797",
    "#7B706E",
    "#BCB0AB",
    "#E16A96",
    "#FFBCD3",
    "#B976A3",
    "#DCA3CA",
    "#A3745C",
    "#DDB3A4",
]


def make_scatterplot(data, color_key, categories_to_show, cluster_to_top_terms):
    altair.renderers.enable("html")
    value_counts = defaultdict(lambda: 0)
    items = []
    categories_to_show = [c for c in categories_to_show if c != color_key and c != app_config.CLUSTER_OPTION_TEXT][
        : app_config.MAX_CATEGORIES_ON_TOOLTIP
    ]
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
                    **{
                        other_category: x["rec"][other_category]
                        for other_category in categories_to_show
                    },
                }
                for group_value in categories_values
            ]
        )

    # Display the value count in the legend
    legend_values = {}  # legend string -> count
    for item in items:
        legend_value = "%s [%0.2f%%]" % (
            item["ColorGroup"],
            100.0 * value_counts[item["ColorGroup"]] / len(data),
        )
        if item["ColorGroup"] in cluster_to_top_terms:
            terms_list = ", ".join(cluster_to_top_terms[item["ColorGroup"]])
            legend_value = "%s [%s]" % (legend_value, terms_list)

        legend_values[legend_value] = value_counts[item["ColorGroup"]]
        item["GroupForLegend"] = legend_value
    df = pd.DataFrame(items)

    # Sort legend values by frequency (descending) so that most frequent are
    # on the top of the legend.
    legend_value_keys = list(legend_values.keys())
    legend_value_keys.sort(key=lambda x: legend_values[x], reverse=True)

    category_keys = [
        re.sub(r" \[.*", "", legend_val) for legend_val in legend_value_keys
    ]
    category_to_color = dict(
        [
            (x, _TABLEAU20_SCHEME[i % len(_TABLEAU20_SCHEME)])
            for i, x in enumerate(category_keys)
        ]
    )
    # Override color for uncategorized to ensure it's grey
    category_to_color[app_config.UNCLUSTERED_NAME] = "lightgrey"

    selection = altair.selection_multi(fields=["GroupForLegend"])
    coloring_scheme = altair.Color(
        "GroupForLegend",
        type="nominal",
        scale=altair.Scale(
            domain=legend_value_keys,
            range=[category_to_color[v] for v in category_keys],
        ),
    )

    color = altair.condition(selection, coloring_scheme, altair.value("white"))

    chart = (
        (
            altair.Chart(df, height=400, width=1000)
            .mark_circle(size=80, opacity=1.0)
            .encode(
                x=altair.X("x", axis=None, scale=altair.Scale(zero=False)),
                y=altair.Y("y", axis=None, scale=altair.Scale(zero=False)),
                color=color,
                tooltip=["Answer", "Categories"] + categories_to_show,
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

    return chart, category_to_color

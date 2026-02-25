import math

import matplotlib
import matplotlib.pyplot as plt

font = {"size": 22}
matplotlib.rc("font", **font)

COLORS = ["goldenrod", "violet", "seagreen", "turquoise", "blue"]


def create_radar_plot(data, labels, color_index=0):
    color = COLORS[color_index % len(COLORS)]
    n_vars = len(data)

    values = data + data[:1]

    angles = [n / float(n_vars) * 2 * math.pi for n in range(n_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})

    ax.set_xticks(angles[:-1], labels, color="grey", size="small")
    ax.tick_params(pad=30)
    ax.set_rlabel_position(0)
    ax.set_yticks(
        [0.25, 0.5, 0.75, 1], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=10
    )
    ax.set_ylim(0, 1)

    ax.plot(angles, values, linewidth=1, linestyle="solid", color=color)
    ax.fill(angles, values, "b", alpha=0.1, color=color)

    return fig

    # df = pd.DataFrame(dict(r=data, theta=labels))
    # fig = px.line_polar(
    #     df,
    #     r="r",
    #     theta="theta",
    #     line_close=True,
    # )
    # # fig.update_traces(fill='toself')
    # fig.update_traces(
    #     fill="toself", fillcolor=color, line=dict(color=f"dark{color}"), opacity=0.3
    # )
    # fig.update_layout(
    #     polar=dict(
    #         radialaxis=dict(
    #             range=[0, 1],
    #             gridwidth=4,
    #             linewidth=4,
    #             linecolor="rgba(255,255,255,0.4)",
    #         ),
    #         angularaxis=dict(gridwidth=4),
    #     )
    # )
    # # fig.update_layout(title={'text': label, 'xanchor': 'center', 'yanchor': 'top', 'y':0.98, 'x':0.5,})
    # fig.update_layout(font=dict(size=43))

    # return fig

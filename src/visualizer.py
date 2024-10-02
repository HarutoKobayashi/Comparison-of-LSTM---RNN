import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize(rnn_reports, lstm_reports, epoch=10):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

    x = [i for i in range(1, epoch + 1)]

    # loss
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rnn_reports["train_loss"],
            mode="lines",
            name="RNN (train)",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rnn_reports["valid_loss"],
            mode="lines",
            name="RNN (valid)",
            line=dict(color="darkblue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lstm_reports["train_loss"],
            mode="lines",
            name="LSTM (train)",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lstm_reports["valid_loss"],
            mode="lines",
            name="LSTM (valid)",
            line=dict(color="darkred"),
        ),
        row=1,
        col=1,
    )

    # acc
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rnn_reports["train_acc"],
            mode="lines",
            name="RNN (train)",
            line=dict(color="blue"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rnn_reports["valid_acc"],
            mode="lines",
            name="RNN (valid)",
            line=dict(color="darkblue"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lstm_reports["train_acc"],
            mode="lines",
            name="LSTM (train)",
            line=dict(color="red"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lstm_reports["valid_acc"],
            mode="lines",
            name="LSTM (valid)",
            line=dict(color="darkred"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        yaxis2=dict(
            range=[0, 1],
        ),
    )
    fig.update_layout(
        title="Results of RNN & LSTM",
        showlegend=True,
    )

    fig.show()

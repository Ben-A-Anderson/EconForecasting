import plotly.graph_objects as go
import pandas as pd

def multi_plot(df, title, col = None,  addAll = True):
    """
    multi_plot will make a plot of all columns in a dataframe and allow the user to choose which to visualize
    
    df = dataframe to be plotted
    title = Title for the plot
    col = list of columns to include in the plot. Default: All columns.
    addAll = Not sure what functionality this offers

    Requirements:
    import plotly.graph_objects as go

    """
    # If no limiting list of columns is passed we plot everything
    if col == None:
        col = df.columns.to_list()
    else:
        # If a subset was passed we only print those columns in the list
        col = col

    # Initilize the Plot
    fig = go.Figure()

    # For each column add a trace
    for column in col: #df.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = df.index,
                y = df[column],
                name = column
            )
        )

    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':False}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': False}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column))),
            pad={"r": 5, "t": 10},
            #showactive=True,
            x=0,
            xanchor="left",
            y=1.08,
            yanchor="top"
            )
        ],
         yaxis_type="linear"       
    )
    # Update remaining layout properties
    fig.update_layout(
        title_text=title,
        height=800
        
    )
   
    fig.show()
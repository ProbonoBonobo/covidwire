import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import base64
import dash_daq as daq
from dash.dependencies import Input, Output, State
import flask
import time
import requests
import plotly.graph_objects as go
import dash_auth

VALID_USERNAME_PASSWORD_PAIRS = {
    "jeff": "CovidWire2020",
    "tom": "CovidWire2020",
    "kevin": "CovidWire2020",
    "loree": "CovidWire2020",
    "mason": "CovidWire2020"
}
#server = flask.Flask(__name__)  # define flask app.server
#server.suppress_callback_exceptions = True
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], assets_folder='assets', assets_external_path= '')

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS

)
server = app.server


app.suppress_callback_exceptions = True
print(dir(dbc.themes.BOOTSTRAP), type(dbc.themes.DARKLY))
print("theme", dbc.themes.BOOTSTRAP)

f = open(".lastmod", "w+")
f.write(str(time.time()))
f.truncate()

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "--primary": "#EF5D60",
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18vw",
    "padding": "2rem 1rem",
    "background-color": "#303030",
    "color": "#f7f9f9",
}

RIGHT_SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'right': 0,
    'bottom': 0,
    'width': '18vw',
    'padding': '2rem 1rem',
    'background-color': "#303030",
    'color': 'white'
}
# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18vw",
    "margin-right": "18vw",
    "padding": "2rem 1rem",
    "background-color": "#303030",
    "color": "#f7f9f9",
    "height": "100%",
}
theme = {"dark": True, "primary": "#6DD6DA", "detail": "#95D9DA", "accent": "#f87575"}

input_group = [dbc.FormGroup(
                [
                    dbc.Label(html.H3("COVID Relevance")),
                    #dbc.Input(placeholder='Enter a number between 1-5', type='number', autoFocus=True, autoComplete=True, id='quality-score')
                    dbc.RadioItems(
                        options=[
                            {"label": "Not at all relevant", "value": 1},
                            {"label": "Indirectly relevant", "value": 2},
                            {"label": "Directly relevant", "value": 3},
                        ],
                        labelStyle={"transform": "translateX(10px)", "margin-left": "20px", "color": "rgba(125, 139, 153, 1)", "transition": "all 0.2 ease-in", "font-weight": "200", "line-height": "1.61rem", "margin-bottom": "0.5rem", "font-size": "1.61rem"},
                        #inputStyle={"background-color": "tomato", "border": "10px solid blue", "color": "chartreuse"},
                        labelCheckedStyle={"font-weight": 900, "transform": "translate(0px 5px);", "color": "#fcfcfc"},
                        id='quality-score')
                ],

            ),
            dbc.FormGroup(
                [
                    dbc.Label(html.H3("Time sensitivity")),
                    #dbc.Input(placeholder='Enter a number between 1-5', type='number', autoFocus=True, autoComplete=True, id='quality-score')

                    dbc.RadioItems(
                        options=[
                            {"label": "High (hide after 24 hours)", "value": "high"},
                            {"label": "Average (show for a couple days)", "value": "medium"},
                            {"label": "Low (show for the next 1-2 weeks)", "value": "low"},
                            {"label": "N/A (this is evergreen content)", "value": "none"},
                        ],
                        value=1,
                        id="time-sensitivity",
                        labelStyle={"transform": "translateX(10px)", "margin-left": "20px",
                                    "color": "rgba(125, 139, 153, 1)", "transition": "all 0.2 ease-in",
                                    "font-weight": "200", "line-height": "1.61rem", "margin-bottom": "0.5rem",
                                    "font-size": "1.61rem"},
                        # inputStyle={"background-color": "tomato", "border": "10px solid blue", "color": "chartreuse"},
                        # inputStyle={"background-color": "tomato", "border": "10px solid blue", "color": "chartreuse"},
                        labelCheckedStyle={"font-weight": 900, "transform": "translate(0px 5px);", "color": "#fcfcfc"},
                    )]),

            dbc.FormGroup(
                [
                    dbc.Label(html.H3("Audience")),
                    dbc.RadioItems(
                        options=[
                            {"label": "Local", "value": "local"},
                            {"label": "Regional", "value": "regional"},
                            {"label": "State", "value": "state"},
                            {"label": "National", "value": "national"},
                            {"label": "International", "value": "international"},
                            {"label": "Unbound", "value": "unbound"}
                        ],
                        value=1,
                        id="audience-label",
                        labelStyle={"transform": "translateX(10px)", "margin-left": "20px", "color": "rgba(125, 139, 153, 1)", "transition": "all 0.2 ease-in", "font-weight": "200", "line-height": "1.61rem", "margin-bottom": "0.5rem", "font-size": "1.61rem"},
                        #inputStyle={"background-color": "tomato", "border": "10px solid blue", "color": "chartreuse"},
                        #inputStyle={"background-color": "tomato", "border": "10px solid blue", "color": "chartreuse"},
                        labelCheckedStyle={"font-weight": 900, "transform": "translate(0px 5px);", "color": "#fcfcfc"},
                    )]),
            dbc.FormGroup(
                [
                    dbc.Label(html.H3("Is this article... ?")),
                    dbc.Checklist(
                        options=[
                            {"label": "Feature-worthy", "value": "feature_worthy"},
                            {"label": "Sports related", "value": "sports_related"},
                            {"label": "Opinion/op-ed", "value": "opinion"},
                            {"label": "Syndicated (e.g., Reuters, AP)", "value": "syndicated"},
                            {"label": "Problematic", "value": "problematic"}
                        ],
                        switch=True,
                        value=[],
                        id="article-tags",
                        labelStyle={"transform": "translateX(10px)", "margin-left": "20px", "color": "rgba(125, 139, 153, 1)", "transition": "all 0.2 ease-in", "font-weight": "200", "line-height": "1.61rem", "margin-bottom": "0.5rem", "font-size": "1.61rem"},
                        #inputStyle={"background-color": "tomato", "border": "10px solid blue", "color": "chartreuse"},
                        #inputStyle={"background-color": "tomato", "border": "10px solid blue", "color": "chartreuse"},
                        labelCheckedStyle={"font-weight": 900, "transform": "translate(0px 5px);", "color": "#fcfcfc"},
                    )]),
    dbc.FormGroup([
        dbc.Button("Submit", color="success", id="submit", n_clicks=0, className='ml-2'),
        dbc.Button("Skip", color='warning', id='skip', n_clicks=0, className='ml-2'),
        dbc.Button("Undo", color='danger', id='undo', n_clicks=0)
        ]),
]
right_sidebar = html.Div(input_group, style=RIGHT_SIDEBAR_STYLE)
def render_audience_graph(values):
    return go.Figure(go.Bar(
        x=values,
        y=['local', 'regional', 'state', 'national', 'international', 'unbound'],
        width=0.618,
        textangle=315,

        marker={"color": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
                          "#e6ab02", "#a6761d", "#666666", "#1b9e77"]},
        # "line": {"width": [0.25,0.5,1,1,1,1]},
        # "colorbar": {"ticklen": 1}},

        orientation='h'),
        layout={'bargap': 0.1, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "boxgap": 0.1,
                "title": {'text': 'Audience Prediction', 'xanchor': 'center', 'x': 0.5},
                'titlefont': dict(family='monospace', size=14, color='rgba(162, 171, 181, 1)'),
                "yaxis": {'gridcolor': 'rgba(162, 171, 181, 0.2)', "color": 'rgba(162, 171, 181, 0.2)',
                          "zerolinecolor": 'rgba(162, 171, 181, 0.2)', "zerolinewidth": 0.1, 'gridwidth': 0.1,
                          'showgrid': True,
                          "tickfont": dict(family='monospace', size=14, color='rgba(162, 171, 181, 0.6)'), },
                "xaxis": {"range": [-4, 8], 'tickwidth': 0.1, "color": 'rgba(162, 171, 181, 0.2)',
                          "zerolinecolor": 'rgba(162, 171, 181, 0.2)', "zerolinewidth": 0.1,
                          'gridcolor': 'rgba(162, 171, 181, 0.2)', 'gridwidth': 0.1, 'showgrid': True,
                          "tickfont": dict(family='monospace', size=14, color='rgba(162, 171, 181, 0.2)')},
                'margin': {'l': 0, 'r': 0, 't': 50, 'b': 0, 'pad': 0}, 'height': 200, 'width': 322}

    )
def render_approval_matrix(values):
    x,y = values
    return go.Figure(go.Scatter(x=[x], y=[y]),
              layout={'bargap': 0.1, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "boxgap": 0.1,
                      "title": {'text': 'Approval Matrix', 'xanchor': 'center', 'x': 0.5},
                      'titlefont': dict(family='monospace', size=14, color='rgba(162, 171, 181, 1)'),
                      "yaxis": {"range": [3, -3], 'gridcolor': 'rgba(162, 171, 181, 0.2)',
                                'title': 'similarity to rejected articles', 'tickangle': 270, 'tickvals': [-2, 2],
                                'ticktext': ['less relevant', 'more relevant'],
                                "color": 'rgba(162, 171, 181, 0.2)', "zerolinecolor": 'rgba(162, 171, 181, 1)',
                                "zerolinewidth": 0.1, 'gridwidth': 0.1, 'showgrid': True,
                                "tickfont": dict(family='monospace', size=14, color='rgba(162, 171, 181, 0.2)'), },
                      "xaxis": {"range": [-3, 3],
                                "title": "similarity to approved articles",
                                'tickvals': [-2, 2], 'ticktext': ['less relevant', 'more relevant'], 'tickwidth': 0.1,
                                "color": 'rgba(162, 171, 181, 0.2)', "zerolinecolor": 'rgba(162, 171, 181, 1)',
                                "zerolinewidth": 0.1, 'gridcolor': 'rgba(162, 171, 181, 0.2)', 'gridwidth': 0.1,
                                'showgrid': True,
                                "tickfont": dict(family='monospace', size=14, color='rgba(162, 171, 181, 0.2)')},
                      'margin': {'l': 0, 'r': 0, 't': 50, 'b': 0, 'pad': 0}, 'height': 322, 'width': 322})

sidebar = html.Div(
    [
        html.H2("CovidWire", className="display-4"),
        html.Hr(),
        html.P([html.Img(src="assets/robot-reading.svg", className='icon-md'), "Machine Learning Dashboard"],
               className='ml-text'),

        html.Div(className='flexy', children=[
        dbc.Nav(
            [
                dbc.NavLink("Page 1", href="/page-1", id="page-1-link", className='hidden'),
                dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
                dbc.NavLink("Page 3", href="/page-3", id="page-3-link"),
            ],
            vertical=True,
            pills=True,
            className='hidden'
        ),
        html.Div(id='audience-prediction', children=[dcc.Graph()]),
        html.Div(id='approval-matrix', children=[]),

                ]
            )
    ],
    style=SIDEBAR_STYLE,
)

from more_itertools import grouper
from spacy.lang.en import English

nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def render_content(headline="None", image_url="localhost:8000/404.png", publication="", description="", content="", audience_filters=['local'], auth="", url=""):
    content = html.Div(
        children=daq.DarkThemeProvider(
            theme=theme,
            children=html.Div(
                [
                    dcc.Interval(id="tick"),
                    html.Div(
                        [
                            # dbc.Row(
                            #     [
                            #         dbc.Col(
                            dbc.Form(
                                [
                                    dbc.Container(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Label("Categories:",),
                                                        width=1,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Dropdown(
                                                            options=[
                                                                {
                                                                    "label": "local",
                                                                    "value": "local",
                                                                },
                                                                {
                                                                    "label": "regional",
                                                                    "value": "regional",
                                                                },
                                                                {
                                                                    "label": "state",
                                                                    "value": "state",
                                                                },
                                                                {
                                                                    "label": "national",
                                                                    "value": "national",
                                                                },
                                                                {
                                                                    "label": "international",
                                                                    "value": "international",
                                                                },
                                                                {
                                                                    "label": "unbound",
                                                                    "value": "unbound",
                                                                },
                                                            ],
                                                            id='audience-filters',
                                                            value=audience_filters,
                                                            multi=True,
                                                        ),
                                                        width=4,
                                                        className="mw-25",
                                                        # style={"min-width": "25vw"}
                                                    ),
                                                    # className="w-100",
                                                    #     inline=True,
                                                    #     className='mr-4',
                                                    # dbc.FormGroup(
                                                    #     [
                                                    dbc.Col(
                                                        dbc.Label("Password:"), width=1
                                                    ),
                                                    dbc.Col(
                                                        dbc.Input(
                                                            placeholder='Enter password here',
                                                            value=auth,
                                                            type='password',
                                                            id='auth'
                                                        ),
                                                        width=4,
                                                    ),
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),
                            html.Br(),
                            # className="w-100",
                            # inline=True
                            #         ),
                            #     ),
                            # ]
                            #     ),
                            # ]), ]),
                            # dbc.Row(
                            #     [
                            #
                            #         dbc.Col(
                            #             dbc.Card([dbc.CardImg(src=image_url)], style={'max-height': '38vh'}), width=12,
                            #
                            #         ),
                            #         # dbc.Col(width=4),
                            #
                            #     ],
                            #     align='center',
                            #
                            # ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Jumbotron(
                                            [
                                                html.H6(
                                                    publication,
                                                    className='text-muted',
                                                    style={"font-variant": 'small-caps'}
                                                ),
                                                html.H1(
                                                    html.B(headline),
                                                    className="display-3",
                                                    id="headline",
                                                ),

                                                html.P(
                                                    description,
                                                    className="lead",
                                                ),
                                                # html.Hr(className="my-2"),
                                                dbc.Card([dbc.CardImg(src=image_url)], style={"margin": "1.5vh 3.37vh 1.5vh 0.57vh"}),
                                                *[html.P(f"""{line1} {line2 if line2 else ""}""",
                                                    className='article-content') for line1, line2 in grouper(nlp(content).sents, 2)],

                                            ],
                                            style={"padding": "20px"},
                                        ), width=11
                                    )
                                ],
                            style={'margin-top': '5vh'}),

                        ]
                    ),
                ]
            ),
        ),
        style=CONTENT_STYLE,
    )
    return content


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        html.Div(id="page-content", children=render_content()),

        right_sidebar,
        dcc.Store(id='appstate',
                                      data={'history': []})
    ]
)


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
# @app.callback(
#     [Output('headline', 'children')],
#     [Input('tick', 'n_intervals')]
# )
# def initialize(n):
#     print(f"n is {n}")
#     if n is None:
#
#
#             print(f"Headline is: {headline}")
#             return headline
#
#         else:
#             return f"error: {res}"
#     else:
#         raise dash.exceptions.PreventUpdate


@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):

    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]
import json
from collections import defaultdict
@app.callback(
    [Output("page-content", "children"), Output("audience-prediction", "children"), Output('approval-matrix', 'children'), Output('appstate', 'data')],
    [Input("url", "pathname"), Input("submit", "n_clicks"), Input("skip", "n_clicks"), Input("undo", "n_clicks"), Input('audience-filters', 'value')],
    [State('auth', 'value'), State('quality-score', 'value'), State('audience-label', 'value'), State('appstate', 'data'), State('article-tags', 'value')]
)
def render_page_content(pathname, submit, skip, undo, audience_filters, auth, score, audience_label, appstate, tags):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    action = button_id if button_id in ('skip', 'submit', 'undo') else 'skip'
    # n_clicks = n_clicks if n_clicks else 0
    print(audience_filters)
    print(appstate)
    # appstate = defaultdict(lambda: "None", appstate)
    _kwargs = {'p': submit,
               'action': action,
               'history': appstate['history'],
               "quality_score": score,
               "audience_label": audience_label,
               "url": "",
               "title": "",
               "name": "",
               "description": "",
               "content": "",
               "syndicated": 0,
               "opinion": 0,
               "feature_worthy": 0,
               "time_sensitivity": 0,
               "sports_related": 0,
               "problematic": 0}
    if appstate and 'url' in appstate:
        _kwargs.update({"url": appstate['url'], "title": appstate['title'], "name": appstate['name'], 'description': appstate['description'], "content": appstate['content']})
    if audience_filters:
        _kwargs["audience"] = ','.join(audience_filters)
    tags = {tag: 1 for tag in tags}
    _kwargs.update(tags)
    _kwargs['auth'] = auth
    payload = base64.encodestring(json.dumps(_kwargs).encode('utf-8')).decode("utf-8")
    res = requests.get("https://kevinzeidler.com/api/classified", params={"payload": payload})
    headline = ""
    image_url = ""
    description = ""
    publication = ""
    content = ""
    url = ""
    docvec = {"approved": 0, "rejected": 0, "local": 0, "regional": 0, "state": 0, "national": 0, "international": 0, "unbound": 0}
    if res.ok:
        print(f"Fetching results...")
        data = res.json()
        print(f"Results received.")

        results = data["results"]
        nxt = results
        headline = nxt["title"]
        image_url = nxt["image_url"]
        description = nxt['description']
        publication = nxt['name']
        content = nxt['content']
        docvec = nxt['docvec_v2']
        url = nxt['url']


    if pathname in ["/", "/page-1"]:
        body = render_content(headline, image_url, publication, description, content, audience_filters, auth, url)

        approval_vals = [docvec["rejected"], docvec['approved']]
        audience_preds = [docvec['local'], docvec['regional'], docvec['state'], docvec['national'], docvec['international'], docvec['unbound']]
        audience_graph = dcc.Graph(figure=render_audience_graph(audience_preds))
        approval_matrix = dcc.Graph(figure=render_approval_matrix(approval_vals))
        appstate.update({"content": content,
                    "title": headline,
                    "description": description,
                    "url": url,
                    "name": publication})
        appstate['history'].append(url)
        print(audience_graph)
        return [body,audience_graph, approval_matrix, appstate]
    elif pathname == "/page-2":
        return html.P("This is the content of page 2. Yay!")
    elif pathname == "/page-3":
        return html.P("Oh cool, this is page 3!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


import time

# @app.callback([Output('changed_at', 'data')], [Input('usertext', 'value')])
# def _(current_text):
#     print(f"Text changed! Text is now {current_text}.")
#     return [time.time()]
#
# @app.callback([Output('locked', 'data'), Output('led_busy', 'value')],
#               [Input("tick", "n_intervals")],
#               [State('last_processed_at', 'data'), State('changed_at', 'data'), State("locked", "data")])
# def _(n, last_processed, last_changed, locked):
#
#     if not n or n < 10:
#         locked = False
#     prev_state = locked
#     now = time.time()
#     time_since_last_change = abs(now - last_changed)
#     new_state = locked or time_since_last_change > 3
#     print(green(f"working: {WORKING}  old state: {prev_state} new state: {new_state}  last_changed: {abs(int(last_changed - now))}s ago"))
#     if new_state == prev_state:
#         print("no state change")
#         return dash.dash.no_update
#     else:
#         print(f"STATE CHANGE: was {'locked' if prev_state else 'unlocked'}, now {'locked' if new_state else 'unlocked'}")
#     return new_state, new_state
# import re
# import json
#
#
# @app.callback(
#     Output("suggestions", "children"),
#     [Input("usertext", "value")],
#     [
#         State("suggestions", "children"),
#         State("window-size", "value"),
#         State("n_preds", "value"),
#     ],
# )
# def _(text, children, window_size, n_preds):
#     if not text or not re.search(r"\.\.\.\s*$", text):
#         raise dash.exceptions.PreventUpdate
#     else:
#         text = re.sub(r"\.\.\.\s*$", "", text)
#
#     f.seek(0)
#     lastmod = float(f.read().strip() or "0")
#
#     dt = time.time() - lastmod
#     print(f"Count tokens called with {text}. Last modified {dt} seconds ago...")
#     if dt < 3:
#         print("Suppressing update")
#         raise dash.exceptions.PreventUpdate
#     else:
#         f.seek(0)
#         f.write(str(time.time()))
#         f.truncate()
#
#     window_size = window_size or WINDOW_SIZE
#     print(f"Loading predictions...")
#     preds = requests.get(
#         "http://8cde28d3f5e8.ngrok.io",
#         params={
#             "text": base64.encodestring(text.encode()).decode(),
#             "window_size": window_size,
#             "n_preds": n_preds,
#         },
#     )
#     print(f"Got response: {preds}")
#     print(preds.json())
#     preds = preds.json()
#     print(f"Predictions loaded.")
#     print(json.dumps(preds, indent=4))
#     preds = [
#         dbc.ListGroupItem(
#             [html.P(line) for line in s.split("\n")],
#             className="gpt2-output",
#             style={
#                 "height": "100%",
#                 "background-color": "#393F45",
#                 "color": "#A2ABB5",
#                 "border": "0 1px 1px 1px solid #A2ABB5",
#             },
#         )
#         for i, s in enumerate(preds["preds"])
#     ]
#     # for i, button in enumerate(preds):
#     #     _id = button.id
#     #     text = button.children[0].value
#     #     options[_id] = text
#     #     print(f"Setting option {_id} to {text}")
#
#     return preds
#
#
server.suppress_callback_exceptions = True

if __name__ == "__main__":
    app.run_server(
        port=8000,
        debug=True,
        # dev_tools_hot_reload=True,
        # dev_tools_hot_reload_interval=1000,
        # dev_tools_hot_reload_watch_interval=1000,
    )
# get_predictions("Dear Golden,\n\nMy name is Kevin Zeidler and I am")

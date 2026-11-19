"""Render the corresponding section of the packaged mathematical reference."""
from importlib.resources import files
from dash import dcc, html


def section(name):
    source = files('acan').joinpath('MODEL.md').read_text(encoding='utf-8')
    body = source.split('\n## '+name+'\n', 1)[1].split('\n## ', 1)[0]
    return html.Div(dcc.Markdown('## '+name+'\n'+body, mathjax=True), className='p-3',
                    style={'maxWidth': '960px', 'margin': 'auto'})

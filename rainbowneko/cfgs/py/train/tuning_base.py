from rainbowneko.parser import CfgModelParser, neko_cfg

@neko_cfg
def make_cfg():
    return dict(
        model_part=CfgModelParser([
            dict(
                lr=1e-6,
                layers=[''],  # train all layers
            )
        ]),

        model_plugin=None,
    )

from rainbowneko.parser import CfgModelParser
def make_cfg():
    dict(
        model_part=CfgModelParser([
            dict(
                lr=1e-6,
                layers=[''],  # train all layers
            )
        ]),

        model_plugin=None,
    )

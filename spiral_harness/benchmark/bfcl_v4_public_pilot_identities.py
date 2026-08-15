"""Byte identities for the pinned public/development BFCL V4 pilot."""

from types import MappingProxyType

BFCL_V4_PILOT_QUESTION_BLOB_IDENTITIES = MappingProxyType(
    {
        "multiple": (
            316_583,
            "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
        ),
        "parallel": (
            171_896,
            "19f51a82eff42e5d62541aa500115a056eb78f437c2ba1f10415fd7c8e5dda84",
        ),
        "parallel_multiple": (
            347_080,
            "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
        ),
        "simple_python": (
            283_274,
            "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
        ),
    }
)

# Raw-row size, raw-row SHA-256, then canonical {id, question, function}
# payload SHA-256.  The final value prevents caller-supplied row coordinates
# from blessing different candidate-visible content.
BFCL_V4_PILOT_ROW_IDENTITIES = MappingProxyType(
    {
        "simple_python_0": (
            613,
            "9208f93fb0939c43255773e90d5f15a6362f5701891cbcb917a505195d89c5e4",
            "d2de2d69860415abdde2ad303938b99aac87b1d70f1bc23383ad6c21f3b868bb",
        ),
        "simple_python_211": (
            842,
            "9dbb32d93e8dd13519d6a0a63556293df40759752b4a59e2d18396585c89974e",
            "c86f9a63e79421149fe490936f4180da7ce7ab9c345c05acc336687bf1fb0e08",
        ),
        "multiple_5": (
            1_780,
            "1fb6b7e29cb1a39fa13cd4ebdd437ecf49226b41f800faa05a5a2297e3d7bd32",
            "ee74249862f1cebb16be693e74bfe5483db5a1bfe0216de810ad9ff19d9f1d10",
        ),
        "parallel_0": (
            612,
            "32090c6b3144973281bb2cd4041f84addb5695d93cfce47ab62757f294a768e6",
            "d8b4e5b271fce60c020e101d69595cfe866cdc92e42049a578aa0ad0101b970a",
        ),
        "parallel_multiple_9": (
            957,
            "e68bc9fb4ce14eb903f09894db7e7668f1d13ff9cca362f3c1c202d3337cecf1",
            "2f06e091c0b4f621f1afd4ef1b7a341e7a5f499c10500e99c38e43fd04c0bfc8",
        ),
        "multiple_10": (
            1_435,
            "dfab531f79f8add12530df2d9f51aa9bb49024887e53f0c36d6666affe6149ee",
            "ac8412758bd7b671ff12b7ac125c933f39dfd68fea60453e654db48d2fd261bc",
        ),
        "parallel_multiple_11": (
            1_155,
            "a960d93439999fce974450606c35da5e88a924057a585f382eb4f436b833bc5c",
            "f7cefa67980dc938b2792cad65ed74c6bcd4d12053e85a78ce586b04ac2d1d24",
        ),
        "simple_python_87": (
            522,
            "a8d194aea0caaa8acc567d28de83ce8fa35ca0c1910506d9446a640f02fcfb35",
            "328a9651ff8ca66ba1dbc2cc6c331e33ebd038821ee8cb4521c9215f63994899",
        ),
        "simple_python_128": (
            719,
            "f30d18ad335053fc11943c04326479882f7a1c29a140d8bd8b246dda2e9594f6",
            "a8c74c7b70175461f1d6c9424f09cd266c63d9d56e7ea49516bb913f906d4e36",
        ),
        "multiple_7": (
            1_398,
            "169f31a12cfe8bf4db2d936cecdabb6b4eed0f61ebc606c9411db20fdce8ea28",
            "841b054161a3ba483dd2f7e50e6fb65fe3d12c41d2d4366ebcb943a9fa7f7b84",
        ),
        "multiple_8": (
            1_578,
            "1ffcd2223a549860af965154f0a12661999c46caaf51a95dc3c00bc412998d48",
            "df19309609b2a16da361324d435e8ebe26256b2f9f3b1ef8dc92c9d2057a5ba4",
        ),
        "parallel_3": (
            560,
            "14186158b593a1a7ac630f355ae17911c62c55225904c6544a50d0448f294002",
            "eea0c86dd64fe4fdbd050d9c6652d5f1d8cc8e47ae2c6bc5ca38d7d90c3f742d",
        ),
        "parallel_4": (
            584,
            "f5d17ddae831c2e8e31e19a925b2c47f6de277a0e342656ed7a433f04a5a9515",
            "74c386c7a94781afde02455c8f546bec2f57167b06279381bc6d982aaaace677",
        ),
        "parallel_multiple_5": (
            1_141,
            "42b8e5c44599672e7ef30f5c4f138292048960605e3c3e585615762ab823ca7c",
            "1e28a25b3bf87bb4fbd8ab8c1fa28c4e39ea86e3b6105463825f3f2beab07d27",
        ),
        "parallel_multiple_55": (
            1_481,
            "aefe928ac663ba3248faee0c8d6bac05b38e83a313bf3a9abc7dbcf354ec10ed",
            "24a88896b149e54086c73bd390a8c29dbc6baf429e27d6a3159ebd9b0784d32b",
        ),
    }
)

BFCL_V4_PILOT_ADAPTER_SOURCE_IDENTITIES = MappingProxyType(
    {
        "berkeley-function-call-leaderboard/bfcl_eval/constants/type_mappings.py": (
            1_813,
            "1702fb67afbe2c492608e58e2b7d02e46381f50166b47f3c952f76e34c7cd3bd",
        ),
        "berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py": (
            33_694,
            "f78fd3edce603b333dc9a88ee2c041dc547d51f71aa449ffebc044c4b1e353f3",
        ),
    }
)

__all__ = [
    "BFCL_V4_PILOT_ADAPTER_SOURCE_IDENTITIES",
    "BFCL_V4_PILOT_QUESTION_BLOB_IDENTITIES",
    "BFCL_V4_PILOT_ROW_IDENTITIES",
]

# ruff: noqa: E501
"""Byte and row identities for the pinned tau-cubed public dry-run."""

from __future__ import annotations

from types import MappingProxyType
from typing import NamedTuple


class Tau3FileIdentity(NamedTuple):
    """One source file consumed by the provider-free prerequisite."""

    git_path: str
    size: int
    sha256: str


class Tau3PilotRowIdentity(NamedTuple):
    """Trusted identities for one selected public row."""

    benchmark_domain: str
    upstream_task_id: str
    partition: str
    selection_stratum: str
    selection_pool_size: int
    selection_rank: int
    source_cluster_id: str
    selection_sha256: str
    row_size: int
    row_sha256: str
    source_projection_size: int
    source_projection_sha256: str
    template_text_size: int
    template_text_sha256: str
    template_token_count: int
    template_token_sha256: str
    source_text_size: int
    source_text_sha256: str
    source_token_count: int
    source_token_sha256: str


def _file(git_path: str, size: int, sha256: str) -> Tau3FileIdentity:
    return Tau3FileIdentity(git_path, size, sha256)


# Order is part of TAU3_CRITICAL_FILE_BUNDLE_SHA256. It follows the prospective
# audit order: environment, data, runtime, evaluator, domain code, then telecom
# task-generation lineage.
# One source per line is easier to audit than formatter-expanded hash literals.
# fmt: off
TAU3_CRITICAL_FILE_IDENTITIES = (
    _file("pyproject.toml", 2_417, "23d59670b4ad7bbc0f57420fd9643a8d9188c24abe4f29c9c965544d8cc4cb8d"),
    _file("uv.lock", 466_363, "62d3a8c4807b89e85703b3c03f2c21048a2da9736ca83d9af6f61adc74ac5314"),
    _file("data/tau2/domains/airline/tasks.json", 155_528, "ccd8ba737b4cc371415af70151187788f728d6108d0916e73bb4317b40542052"),
    _file("data/tau2/domains/airline/split_tasks.json", 1_443, "b22ced4d9a9850ac9aea31c53bdcb6d6009058140bd9acc7db37c1d36222ba8b"),
    _file("data/tau2/domains/airline/db.json", 7_042_725, "1af9fea6e03ca7ca15a22bb3fcaf3e351393e3fc9070b6777947da8996f7531b"),
    _file("data/tau2/domains/airline/policy.md", 7_676, "10dc0525421521208be39cee235bba84a16e2bcba9899eb93d92cd81d2f62fc4"),
    _file("data/tau2/domains/retail/tasks.json", 345_982, "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8"),
    _file("data/tau2/domains/retail/split_tasks.json", 3_263, "ed0580ec52575b63fbf76568af42490da6ee7783ecb4aa81af46961291358f20"),
    _file("data/tau2/domains/retail/db.json", 2_811_616, "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765"),
    _file("data/tau2/domains/retail/policy.md", 6_699, "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467"),
    _file("data/tau2/domains/telecom/tasks.json", 13_977_063, "37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4"),
    _file("data/tau2/domains/telecom/split_tasks.json", 356_149, "605b488bb9a6acb3c7f4505240a855fdc8681d09aadb16a8f38b2efcfc5c3aec"),
    _file("data/tau2/domains/telecom/db.toml", 9_628, "562d647ef9d7df8df91eafd8ee76036e707c8f9e32aaedc4a7fde06975aea2c0"),
    _file("data/tau2/domains/telecom/user_db.toml", 933, "4886107ea0c16f8e16d74bef91cfb557de61822afe9361f5a9b241806aea2e4c"),
    _file("data/tau2/domains/telecom/main_policy.md", 5_699, "95943844a0cf11fdf0e2842b483b81d9f9338aa53235712ed131db075d086ed7"),
    _file("data/tau2/domains/telecom/tech_support_manual.md", 17_589, "015a3ee49ec8c199c5e1a059c922081937b92cf3a5ea74fedbc4357816f389ba"),
    _file("data/tau2/user_simulator/simulation_guidelines.md", 1_449, "740a29dfa64d7bc08eea3bf7493575b914a63f744acbaf7f199ee07eddaf72d3"),
    _file("data/tau2/user_simulator/simulation_guidelines_tools.md", 2_747, "cbf3d8a4d8642fd04e559862f1afef55d7dd4e6a7e727ca49e239023c599de0c"),
    _file("src/tau2/config.py", 9_952, "1f3589ff61392778079d688ad18b680ce13275a84462b2383b258614b665eb46"),
    _file("src/tau2/registry.py", 13_109, "d170c1f4b7323a75e26b2ead6754ed423b17a3deb91fe2c89e7e8ec78c6bf12a"),
    _file("src/tau2/run.py", 10_098, "5a32c35582b8070e8d50c135508d436859392bf2db179a52711a3c53fb108dc9"),
    _file("src/tau2/data_model/message.py", 32_440, "378cc451ded5b4314c4fe177bf83e72abb2e0bba53b79c2a0e27c53f3cbd15b0"),
    _file("src/tau2/data_model/persona.py", 5_854, "e012387070438f51673aed61291288541c7270bce310b9851fde89493fac8e19"),
    _file("src/tau2/data_model/simulation.py", 58_699, "518a4eda9af783f25243e1bd2c714f4617cf86188878a0bb584fe73c9528a35a"),
    _file("src/tau2/data_model/tasks.py", 22_764, "0243a6454b2e9c8a14b8364c38ff95dc1dd7512105b69fada9bf9295fc8170fa"),
    _file("src/tau2/environment/db.py", 1_267, "49e12f900a8aabe62d6137a9baa1e7d471f9b3e516cf7c85fb61ecde8a65154c"),
    _file("src/tau2/environment/environment.py", 18_486, "b5357f11e2914c11ce0372af919cc1d726191b91cd97f776b8fb388ae0615350"),
    _file("src/tau2/environment/tool.py", 8_092, "6dd74021ba95a6f8dfa0971546b768b90e15b212f8f9051156ae6bcdd24b77d3"),
    _file("src/tau2/environment/toolkit.py", 11_313, "3e897891e8878c3399aec72e42eb0eb5546642420c0c902b225421fc936b924a"),
    _file("src/tau2/agent/base/llm_config.py", 1_276, "3b5bf4c5f69e5df5357374a9e379e6932096bcdbfcf3e411c4791c33d73d35b4"),
    _file("src/tau2/agent/base/participant.py", 8_137, "64609023945372948cbec447fbf3b9d7b2eb53701a185680305af5247402d420"),
    _file("src/tau2/agent/base_agent.py", 6_331, "97f1033723d03da99f3338ff7a12ed60114a55ed293928ff4ca0bdace6851b4d"),
    _file("src/tau2/agent/llm_agent.py", 18_019, "d13412ff502859f0ef0492af0128f1b2666be32640057ba8e076d2602af084d8"),
    _file("src/tau2/user/user_simulator.py", 9_350, "bfe1defe73bce25450b2be697c249d6cec2f3ba25a70e32e2452ba75a53a630c"),
    _file("src/tau2/user/user_simulator_base.py", 6_622, "e6c4bae08fb7ae6fa6ee5ee29624bc35e0dd6c6fe8ffeab065592c7b51fa32f2"),
    _file("src/tau2/orchestrator/modes.py", 808, "b2d131cef3f15b1558568b1e4b0b316aeea26ae7fbe1e1092d79febbb7d74553"),
    _file("src/tau2/orchestrator/orchestrator.py", 40_843, "6c689e5d1037dd08aef8834b46f58efa14359824cbebf7084de0172e5f580cba"),
    _file("src/tau2/runner/batch.py", 33_414, "649a73d5cd4b373abfaca2bacfcc97693a43da4ef1dbb6b9098f673c7a0161db"),
    _file("src/tau2/runner/build.py", 21_425, "2c4b61b83d4108bb9a2b46724a7b9cc0823548cfd866ca0f415bc536269d964b"),
    _file("src/tau2/runner/checkpoint.py", 16_355, "3c4cf66101ec30a3dc47ad7b3be35a4e542ffcbf85b3ba2bad66ad28064e9ca3"),
    _file("src/tau2/runner/helpers.py", 6_851, "39f804b325b8bb73893fc75ba422a70d5bacd258bc64ebaf1c0fe71b2fa93b66"),
    _file("src/tau2/runner/progress.py", 7_868, "3264ea8ad797f6d513626b9baec1a57c2a3b0aa499cebf9915f9038b90edf68f"),
    _file("src/tau2/runner/simulation.py", 3_240, "1b98532199a05d87dcc14aead9d59c320d20f11d3d8e4c8458663ed5964eafef"),
    _file("src/tau2/utils/io_utils.py", 5_890, "ce66fb275b9bbf09a17be2cb2092bf2e696ab07655bf7571fe53ba8db4ed3015"),
    _file("src/tau2/utils/llm_utils.py", 16_869, "8c1f1c297e879b2a2d48ac65394cdd497bbd10f42fefa0ce770e7a65ce42bf95"),
    _file("src/tau2/utils/pydantic_utils.py", 994, "618dfc39b619bb0f623151682962ddc52bf2ea70bc2562dded884f118c84b78a"),
    _file("src/tau2/utils/utils.py", 2_957, "e0c39544b01700e60a3a26c7cf2c2d17e37e3852d5f0fa466b8517f82b83622b"),
    _file("src/tau2/evaluator/evaluator.py", 14_218, "204d95a82812402f2fee989250392d3c66de702a8bcc20258c7e2a8153a1ea8d"),
    _file("src/tau2/evaluator/evaluator_action.py", 8_150, "4347ec90fd09db73b9f7e46515220363f91f0c3b12aeddf7ece83346f49227f8"),
    _file("src/tau2/evaluator/evaluator_base.py", 665, "1f79c11e5cb5c8cabc26cdaad7ff3a3a1687380f4d7fb1f9fbd9b6f9ab18e1ff"),
    _file("src/tau2/evaluator/evaluator_communicate.py", 7_462, "7a082f91c746b84cc37c6994eb6ff78f297222fd067503a4cc86d45750696507"),
    _file("src/tau2/evaluator/evaluator_env.py", 14_100, "b53ebfe6b0b06d7a2071b5f9dd03ee6ca8ac6000b0ef44f3bd345b346c30d8ac"),
    _file("src/tau2/evaluator/evaluator_nl_assertions.py", 8_843, "cb6b3758cdceb02a726732b4ee1af618affcc08f8ab84b44f060146b8f3ad90a"),
    _file("src/tau2/metrics/agent_metrics.py", 18_374, "136b5ebcc0c115b5817a4f6d6b485a871aba199ddedf56458f35f339569a54dc"),
    _file("src/tau2/domains/airline/data_model.py", 10_402, "a8083e2f590e4b9b66a65d7cd50b7a0247e4fa33d6961aaeb02e4d92488b3cff"),
    _file("src/tau2/domains/airline/environment.py", 1_612, "64589faffbcb75c8ac7c95d2c94ad27f4bd70e23b9856a6928fa678a0f7fcc8d"),
    _file("src/tau2/domains/airline/tools.py", 28_123, "3987f21c286314cb48764f97052934ff4fd60e27a0b2e4a43423911adf8eaa26"),
    _file("src/tau2/domains/airline/utils.py", 256, "9da26327af3d7fa94e05d48869d8b06cd8110313fc24d308993a750106b11f06"),
    _file("src/tau2/domains/retail/data_model.py", 8_799, "f5ad09590953855cc49d2ab2ad5c0594be408cae74d1500fb278e5a6bf2c50d6"),
    _file("src/tau2/domains/retail/environment.py", 1_615, "5b819fb895a4ef49df71404c3d9678ab09e8578eec841b8ec9229c8f51e32405"),
    _file("src/tau2/domains/retail/tools.py", 29_202, "34f4fe9702d36f0dd4697700fbcd63d0cf5386e6eb8e7499aaa336243bad8750"),
    _file("src/tau2/domains/retail/utils.py", 248, "a385b085b63d0e5921c15f7e2201b00df1d984990f4dbb7b24a867e3bf7cb648"),
    _file("src/tau2/domains/telecom/data_model.py", 9_418, "f868cbbd344cadf6c376f21976edfa38b8cfe199922748e31eee666f791a0628"),
    _file("src/tau2/domains/telecom/environment.py", 7_286, "7bc335d717a8cb6bda54551fa2b5c12ff073f6287878d215f4181572e551abcd"),
    _file("src/tau2/domains/telecom/tools.py", 26_442, "388552434d1b8225e33e7958bdb6efccfe92b49e11d1838568d9b559514e9d01"),
    _file("src/tau2/domains/telecom/user_data_model.py", 15_172, "27be0006694e9a740dd378f42099d6d1afecbd445f7906aedcd1f800087377c2"),
    _file("src/tau2/domains/telecom/user_tools.py", 48_531, "03fa751eeea3734313a3f5274223d42750fd5c1a28f2624265a678f271ac309d"),
    _file("src/tau2/domains/telecom/utils.py", 1_260, "9195707fca3d96b41d78de661bd2408cfaab75793769c9a739a7521b690fc6dd"),
    _file("src/tau2/domains/telecom/tasks/const.py", 2_204, "1f7100229a56c86ea2f47d6c78fe26f7e0fbab65ad91150f9436f0fee09d7784"),
    _file("src/tau2/domains/telecom/tasks/create_tasks.py", 3_639, "7ce75c5afc869a344985e68f7725d18709e51f6aa1233a91e5d387ba45bfac59"),
    _file("src/tau2/domains/telecom/tasks/manager.py", 8_836, "ba1fe1c514e8a85c58c417cfbe5ecfa853289bbc181874e37e90e610bdee3e9e"),
    _file("src/tau2/domains/telecom/tasks/mms_issues.py", 10_313, "f7350a6771ac28464ee392731f741486b421126ddf907082891d2cfc4a739b24"),
    _file("src/tau2/domains/telecom/tasks/mobile_data_issues.py", 16_347, "2c5169d82c6fad894ffd9447b98c249312155354d428fb3d3f373176b82fd667"),
    _file("src/tau2/domains/telecom/tasks/service_issues.py", 14_396, "a646ab516cdd7f9a3716dfa9acd7ca60741bb7db81494c26c2c988f2efe3db2a"),
    _file("src/tau2/domains/telecom/tasks/utils.py", 4_162, "aaf9ee7d1e3699918292902951ca929ff7d0e7d7a2c8f782a0bd14e3bf13ca26"),
)
# fmt: on

TAU3_CRITICAL_FILE_BUNDLE_SHA256 = (
    "25097699bd2c87a914738dd4f4e350973189cb702b16d44ab874ffe609185cd5"
)

_COMMON_TOOL_PATHS = (
    "src/tau2/environment/db.py",
    "src/tau2/environment/environment.py",
    "src/tau2/environment/tool.py",
    "src/tau2/environment/toolkit.py",
)
TAU3_DOMAIN_POLICY_PATHS = MappingProxyType(
    {
        "airline": ("data/tau2/domains/airline/policy.md",),
        "retail": ("data/tau2/domains/retail/policy.md",),
        "telecom": (
            "data/tau2/domains/telecom/main_policy.md",
            "data/tau2/domains/telecom/tech_support_manual.md",
        ),
    }
)
TAU3_DOMAIN_TOOL_PATHS = MappingProxyType(
    {
        "airline": _COMMON_TOOL_PATHS
        + tuple(
            f"src/tau2/domains/airline/{name}.py"
            for name in ("data_model", "environment", "tools", "utils")
        ),
        "retail": _COMMON_TOOL_PATHS
        + tuple(
            f"src/tau2/domains/retail/{name}.py"
            for name in ("data_model", "environment", "tools", "utils")
        ),
        "telecom": _COMMON_TOOL_PATHS
        + tuple(
            f"src/tau2/domains/telecom/{name}.py"
            for name in (
                "data_model",
                "environment",
                "tools",
                "utils",
                "user_data_model",
                "user_tools",
            )
        ),
    }
)
TAU3_DOMAIN_POLICY_BUNDLE_SHA256 = MappingProxyType(
    {
        "airline": "1edfe3b65ff0642e244041f980d33b6683c945e3beea5daaea9aede1e78841c5",
        "retail": "92c2d27f57fad8eddd3666cf4c8580708ec99a2dda2e6170a0f52bb89ebc9113",
        "telecom": "8f15a18b526b69eb3686cf0b83732ef176a44c01a88816bf56476ee28f49ecdc",
    }
)
TAU3_DOMAIN_TOOL_BUNDLE_SHA256 = MappingProxyType(
    {
        "airline": "21f4afdb3582373b81daa319489398da5c7512d1364e3c9981c5f3abe1a685ca",
        "retail": "609fc14bc48be31b01b29fe1ce388689e8d43e4bed7ecd23b5db5e1733bd3872",
        "telecom": "7281b4e10274e91607560e59ebcce65cab62946b0022f4025cf0887fde5eb076",
    }
)

TAU3_PILOT_ROW_IDENTITIES = MappingProxyType(
    {
        "tau3p-635ce58fdabd481b": Tau3PilotRowIdentity(
            "airline",
            "5",
            "fit",
            "airline-base",
            50,
            0,
            "lineage-unresolved-airline",
            "038bdad770146dd40998398e43cec8eaca98ff300ddc2abd114c8c1c8af7fa93",
            1_461,
            "e16b47dd111fe257cfeb2dc158975b196419f75e7de4d273be11d8b88154ce33",
            1_040,
            "65bffa977efbd9262d894d881949c523c545a5c8cda58f51f5b0284a33c9d5ee",
            749,
            "530c39a9ce994b23eb500536876c26ce8d5572e65a74b5e8670759ab2e7a14ab",
            56,
            "017c1aab3bb0dfd89625321981e502de5ad06c6c3a99a8fdeaaa1b656fc9cbd3",
            51,
            "e07f6b425c26c98af960cf049f8da01f8bbff42a7585b3925ebf96d543160c8c",
            2,
            "1555f51ff1812833979c16f76436b203fa16d0ae51b944345f30572b8e34a3eb",
        ),
        "tau3p-cb806d5de2884063": Tau3PilotRowIdentity(
            "airline",
            "7",
            "fit",
            "airline-base",
            50,
            1,
            "lineage-unresolved-airline",
            "0880ae84f8cb54d2934527384bdcc5f7be8167de5c070d62a51bea2d9f1e0838",
            1_883,
            "8c66a0f45bbb05bfd11a8444a417d26e091bb95dc9f7e309f20806cf49f17e2e",
            824,
            "e9f98b54a2339f21f11a12cc5c90b391a0124fcf8b66f028f6228eea1f78a84c",
            558,
            "47c72a3c7dbd9f549609300c14689abe6e6556d33ffeabc26e8e3bd4b506728e",
            46,
            "2b28027b804cb5b3d1c6f1a90c981aa168675e5ed5f0100d2925bb1792636aa4",
            36,
            "c2dd257026258261decf4534f491cc5ef1eb54d58694a831c987c61d6889d935",
            2,
            "0dabbc1c9ee9f1767651858577b8b1ff100ae1cc0008679ae8ad1575930db928",
        ),
        "tau3p-f1033b915ea74be1": Tau3PilotRowIdentity(
            "airline",
            "38",
            "gate",
            "airline-base",
            50,
            2,
            "lineage-unresolved-airline",
            "1f462ce0d99ebe24f370d9c4fd0251f278a408c9d0c8d1292bb731156e98decf",
            1_660,
            "2da568178256a77e0d48f93351151b212cb3c0ba77dfb2f81d2922d5a7ef2c35",
            1_092,
            "7e4988d3dab07a3735cba62babaeed9482fdcbfce42439d623574109c437efc9",
            769,
            "2978d72964ebae6186f3c1727c709badc5e0d6735aa526800cc765ff04b33525",
            45,
            "c676c85252e8531e2ac5a8c910cae23cf282eb7278f579ed1cef5a0a74d4683d",
            54,
            "dfcfa0fd7aa6e162110d399684e9ca596f3f58103c6bccb3879e887c266adfe3",
            2,
            "bf722a30b17f7ffb9a0d2249c833b77bcb06cbacff568ad86a0a0ce066aaca8c",
        ),
        "tau3p-7e42d1a33f114910": Tau3PilotRowIdentity(
            "airline",
            "29",
            "gate",
            "airline-base",
            50,
            3,
            "lineage-unresolved-airline",
            "1f4dbac69228db9eef32637a4dcc10d96cfbf96accdc41e6069cd9a6f6750b14",
            2_326,
            "3bd8a653a5e707509efc50fce25d638df527fa9debdce81660d28ee3214c79cf",
            1_259,
            "fe0af9b41c18a6cafab63e4170620d39a088bc5c7edeb7085a9e2d60ab0d92b9",
            909,
            "73ef29d261a956a2a6ba52207ed50ea5df1e48718ba2faa3976efbef6438c0a4",
            65,
            "558fe7aac62c2fa9efa0efc0fb3f66e070ea8ae2f62642c86993749a96e8e988",
            107,
            "401373ae111528eceec0350f4865a13ad544909389826aaa2f582c702452bbe7",
            7,
            "109b63f0608b9e45d0fd0631ec4634d6816cb1d6a3b673984bc27916d470f7b3",
        ),
        "tau3p-1399f08cea4e4ca8": Tau3PilotRowIdentity(
            "airline",
            "23",
            "holdout",
            "airline-base",
            50,
            4,
            "lineage-unresolved-airline",
            "2397924fc62d085cc4307d506ae6559c2c64b5cf670dfc4b6e270c790dea6ee1",
            4_745,
            "aef15892125b9d26bffc6d65e1d5baf42d8b6a941a9fd0492346582317605f37",
            1_806,
            "7800495875e9e3f0ce313ee3a36fc317bb52d62fac49563c34479d9aed01d46f",
            1_503,
            "bc5e32a5ffe34cba53aacd4ea6edba661b0d56344fd12576e2612aa31c16b89e",
            88,
            "13d02713880fb45659457ed50caa0865d6f22503246b492f7a5e4ba940192abf",
            58,
            "a05ac9895e0d538af10527fad97f54c436cae7c3fc5fdac67343847581f1a784",
            2,
            "085c39a1a3af6e83d9c69bcb204e964aa1ae5a7a8b43e57391bc715be1bf8fac",
        ),
        "tau3p-71909394dbb4431f": Tau3PilotRowIdentity(
            "airline",
            "35",
            "holdout",
            "airline-base",
            50,
            5,
            "lineage-unresolved-airline",
            "25de49c0b37b8bd957675369cfd66cfb793b56c2e94bc0e590d1fdab56c18f4e",
            1_977,
            "719313458694d16d0d52bf87b52a70fc33fa41e2df3740658f401e6b0099df49",
            1_112,
            "ca73e924faff2554c44b505ed810b35c648dee019d21eebb2ab44e09a5807b3a",
            818,
            "1e1eb1c1137408bb726d3fcc8cf93914b5f110aa287a5f90acbaaac68d4381f4",
            56,
            "039bd88c09d26c2c6934db8d3a0d2cf47eea2d498d42891edb24595da151d8d4",
            54,
            "65f3b48b55d5a4d4b91eaa77133f91d1e169e441e25fd64fef20855f8d023f99",
            2,
            "f2ed7247ac7218148d7f34d185ad2dc410295da8da1352a88b0c56eb77aba767",
        ),
        "tau3p-7f5bc4ad03a5494a": Tau3PilotRowIdentity(
            "retail",
            "32",
            "fit",
            "retail-base",
            114,
            0,
            "lineage-unresolved-retail",
            "07b97be3bfd2d2a1f07813baa7c345f6b3e116c44b30069daf8e0542576d1e46",
            2_356,
            "972655ba96045a2feba61c6de8c1c21cc4c2a251712bee347b345ebee36ac9b9",
            756,
            "adfd786ff9a2ac05c930ca6e567b07b2148166e098d75df27da8a594d4d41271",
            447,
            "1dbfd10c53e68d08f1a82a2612d4dccf7b1b96ddfe364369fe8af7d197cea8cc",
            36,
            "215be1f1a9d4b06c9091a4eb67155d84cbf6a71a0e64a89cae90395ab0ca606c",
            61,
            "5dcb6fd9b7f7f5476c8393051755db351ef83db283aef2ff0207f75da02cf0fd",
            4,
            "25cef95c88044b1f00b40dabade27e494bb372fd43ea77ec69dbcd725bd81332",
        ),
        "tau3p-bf591c807eaa42d3": Tau3PilotRowIdentity(
            "retail",
            "41",
            "fit",
            "retail-base",
            114,
            1,
            "lineage-unresolved-retail",
            "0a3ed170ed92b6531f210150316fa8704f22a1cc3e33a8b0e0a98762aaf96183",
            2_523,
            "486e97c520b0aa96b7f8743a11043de3db685cf35f5386ce1045a39077c5ccb2",
            896,
            "f98fc9130e795b589080602c1b184a23746f8fda89cb6a9605ae192d43b65f1a",
            541,
            "04a5fd6e36d5ea2eb85f2a539b8fe37b726482eaf2948004e6affb376b009770",
            48,
            "4be849b5f1140da3ef240f455feba5b73dd408dcc4836e380818f226480aeafa",
            93,
            "05198eb46632536bc03f71c20338330f46cbb878991f6746c475d81564bbdfb9",
            8,
            "92c5236348123ce85d6f386677813a329b0629bb16543ed1b9861b3d41e5fb18",
        ),
        "tau3p-ce4402047ad147ff": Tau3PilotRowIdentity(
            "retail",
            "86",
            "gate",
            "retail-base",
            114,
            2,
            "lineage-unresolved-retail",
            "0a48563dbbbdd4a6f2a04d52616bd69bfa00c3474ad8dce1266f402d9163c375",
            1_100,
            "05f14fc62f7951f3dc21c11d96a976edd59d7c5f9ee11568dc6d5f5c67a012c9",
            540,
            "dfb8afd6676470ee7aecdae5240d3ad6b32fc8be07470e958ed620b2886ab8cd",
            236,
            "39e9df3243662d484f3593ace841b98465141f01b635fa0ec689608a5ff7f14c",
            19,
            "91b696cce6182e8ad34bb2bbaa42511a01e0a21b56717f3b6cb45abdc44e6182",
            78,
            "41bfd7fea56dd3db390c0ae0124888eb140645a83ac0a15c4ee5c03f26b7c1bb",
            2,
            "e064a147d5e7a28a68899c6428137690ecc3f63072e8f2f0ddcf79e78dfec2ed",
        ),
        "tau3p-ccddf871db01464a": Tau3PilotRowIdentity(
            "retail",
            "77",
            "gate",
            "retail-base",
            114,
            3,
            "lineage-unresolved-retail",
            "0ee59f35648ce6dc6e727344ee3dd52c39701a90b785e11183de309f009b7358",
            901,
            "d053ad616c35ba0373f9e546706576ddd84cc90551a9f1a93e26b7e85e6f7e08",
            561,
            "1d9bac2b0587dee1accb803f7d122cdafd7e3fb5af318b4005919a7c8a22af2b",
            247,
            "a6bb56ba1e6fbc2d42621322fe23262b6858f406c7cb2fd65312b1f695b9dd16",
            23,
            "12c9250da0648157238fcbb98a797721281f3d468b77d696a211cb2011d89eae",
            52,
            "6b815b3dea9b4ee544bad97852715e7a536fc069446b51961c20840dd8d4c371",
            2,
            "29c8d920651a0231c200361c5cb09ffba4857d07415ffbc82f22d054f8d00cdd",
        ),
        "tau3p-22852e721395489d": Tau3PilotRowIdentity(
            "retail",
            "95",
            "holdout",
            "retail-base",
            114,
            4,
            "lineage-unresolved-retail",
            "1253ded1e74b7c66b1aa30869eefab09fb436284347258a73ef5d69136da14de",
            1_446,
            "0c364cfbee4d76573ce8ed4e001416ca3d3b431782c78934c006f5f79105e144",
            612,
            "4d7026d44b25a1d440c41409912c51aaffb575f47f46c8d445a17bb5b2577228",
            299,
            "2f8e7e051dc621040390cfbcc66fb9d4eca44571cfa59ad083b500b6cccb1b9c",
            24,
            "f9eb0f9e3812fe17ee5006b5c04f2ac368d18c9cb8352eb0469f9fdfe4903bca",
            50,
            "16ef2b19ee30779bbf45ce1ef9f9185ff07f0f2d73ce9e7a6ea47a176cf8a83a",
            2,
            "02cceee64702edc11c8d16f194a36b86416ed52ccb313f0c0e2c0a49f037e70d",
        ),
        "tau3p-02b26e459dae4959": Tau3PilotRowIdentity(
            "retail",
            "22",
            "holdout",
            "retail-base",
            114,
            5,
            "lineage-unresolved-retail",
            "14e94806ab4e6880461ff457b49a0a131954276958e35ddc05edc40e0e727887",
            1_892,
            "968d73ce2d67f42ca10f1fd2b4368ac05db7bdf842bfcc26ca1afe1fec221b7f",
            693,
            "904bec7ca2f0358e39fb0749d246ea31e860214c4dd8e692810fbcf6c29f0d58",
            379,
            "ecb37242c4c569fb088339bd8a1fa20bf0f3e6a5210bcaa024beb5db39b483e0",
            27,
            "28e2c758a0ecaa7a7561715769d4638e3cd01d4bb304cf215af47e6fe15e672a",
            52,
            "4ea92b9d45e161c87b25e2e1df0e24a2b3a2088e095986d6718e65bf486a2de1",
            3,
            "b59761d48b853134bf08dc06de9d9dc163f491d44d7a8c843d343f624869b96e",
        ),
        "tau3p-983712d857d044c4": Tau3PilotRowIdentity(
            "telecom",
            "[mobile_data_issue]bad_vpn|data_saver_mode_on|user_abroad_roaming_disabled_on[PERSONA:None]",
            "holdout",
            "telecom-mobile-data-base",
            36,
            0,
            "telecom-john-smith-shared-fixture",
            "00a9bf2e754c6668f3fd9f07dd575e228e63b408aff36292a7b96a6fa3a4370f",
            3_846,
            "40bdfe6766901de423000c3522470209d5165d2b017cc413e5ee6313afde6b38",
            2_332,
            "3ae0f3641d007c5a5bc80bea08696287f65bfb2485ec98d31d0939101ba9aedc",
            1_489,
            "cd52a80b064e32b975d36133961a857103e66b602a6eae37f81ea89fc8bc3ae5",
            88,
            "75d273121a04918a88a838eaed92819c8c5c9f83065787be726b565610b5a2e8",
            86,
            "31eab46aee40becca95cb1cb2ccffd5fd5ec3ecad1e84819ad993cbe268bb964",
            5,
            "d173d260d0b645dddaa606df4a4c336e66ac5d8e026a0f8c870a412fe797ee64",
        ),
        "tau3p-fb510248e09f4e52": Tau3PilotRowIdentity(
            "telecom",
            "[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|lock_sim_card_pin|unseat_sim_card[PERSONA:Easy]",
            "holdout",
            "telecom-service-base",
            29,
            0,
            "telecom-john-smith-shared-fixture",
            "0399a5e4c29a4203ea235d9e98570b8c432e488745eb1b3965ee51bbf6af45ce",
            3_804,
            "2fe941ec55028aaf7dc838f540f31a093d1ca6089daf3c2e07780f9a1058311e",
            2_618,
            "d13e7b37e80b683a3b0434821b4adf9911831c1836424e94aab05009d3d29724",
            1_188,
            "a8835f3f02b9b1004c71c853f1fd7fdeedb72b9680aae96c85fadc85ad200756",
            71,
            "4ceabe896d72ee3b268e0d05cea0941d2cda5f060e80308e9a07a5bb2f42ca37",
            50,
            "4a9a9863a82966d45a3aac6ecc8e91525add1ffbb57fae15d9915f90b9d9d3a0",
            2,
            "05e809eb7b68106642837be88861a64bfd8f2a5aea2ada775c24f1a364a89d9a",
        ),
        "tau3p-f115420413bc4901": Tau3PilotRowIdentity(
            "telecom",
            "[mms_issue]bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card[PERSONA:Easy]",
            "holdout",
            "telecom-mms-base",
            49,
            0,
            "telecom-john-smith-shared-fixture",
            "0582964f0662436fa887097f88543e7b953ea29143a742f2519465ccd12498bf",
            5_082,
            "50eed41fbd8711f9e00ba6e3c38a420f74ff8a8ad7d3241c5a45e9af4ac8fb0c",
            2_643,
            "74d02ae7149a7e4185832ef7e7c9908f320329fa97586e5003832866c77876f7",
            1_173,
            "efccfcb97b11b62074ab8162b7f74d4a70872812f4d8d69426cd94e2fa3223ca",
            74,
            "09988499ade81bddfc673c9374c9ac72ec16e31636dbaaee8fbb20a491634c31",
            98,
            "79d42e8cab9482d980e4c5c9da145129a6c06c4e70d2413e6c8280916d87c42d",
            6,
            "24d6f009f2628ec9caf9e52767f8ffe1faec4eeab272f83ffa633e89f2c18cad",
        ),
    }
)

__all__ = [name for name in globals() if name.startswith("TAU3_") or name.startswith("Tau3")]

"""Policy metadata must not enable or overstate reduced-precision execution."""

from dataclasses import FrozenInstanceError, fields
import json

import pytest

from lacuna.errors import CapabilityError
from lacuna.numerical_policy import (
    NumericalCapabilities,
    NumericalPolicy,
    NumericalStatus,
    numerical_policy,
)
from lacuna.precision import PrecisionProfile, require_supported_precision


def test_default_preserves_float64_without_claiming_new_certification():
    policy = numerical_policy()
    assert policy.precision is PrecisionProfile.FLOAT64
    assert policy.preserves_existing_float64_settings is True
    assert policy.required_certifications == ()
    assert policy.authorizes_execution is False
    assert set(policy.capabilities.to_record().values()) == {"existing"}
    assert require_supported_precision() is PrecisionProfile.FLOAT64
    assert set(NumericalStatus) == {
        NumericalStatus.EXISTING, NumericalStatus.IMPLEMENTED, NumericalStatus.UNCERTIFIED,
    }


@pytest.mark.parametrize("profile", tuple(PrecisionProfile))
def test_policy_is_deeply_immutable_and_json_records_are_detached(profile):
    policy = numerical_policy(profile)
    assert hash(policy) == hash(numerical_policy(profile))
    for name, value in (
        ("precision", PrecisionProfile.FLOAT64),
        ("policy_revision", 2),
        ("authorizes_execution", True),
        ("required_certifications", ()),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(policy, name, value)
    with pytest.raises(FrozenInstanceError):
        policy.capabilities.root_finding = NumericalStatus.EXISTING
    record = json.loads(json.dumps(policy.to_record()))
    assert record == policy.to_record()
    record["precision"]["arithmetic_revision"] = 9
    record["capabilities"]["root_finding"] = "certified"
    record["primitive_binding_scope"].append("EXP")
    record["required_certifications"].append("mutated")
    assert policy.arithmetic_revision == 1
    assert policy.capabilities.root_finding.value != "certified"
    assert "EXP" not in policy.primitive_binding_scope
    assert "mutated" not in policy.required_certifications


@pytest.mark.parametrize("profile", tuple(PrecisionProfile))
def test_policy_identity_includes_profile_arithmetic_and_policy_revision(profile):
    policy = numerical_policy(profile)
    assert policy.policy_revision == 1
    assert policy.arithmetic_revision == profile.arithmetic_revision
    assert policy.cache_key == (*profile.cache_key, 1)
    record = policy.to_record()
    assert record["schema_version"] == 1
    assert record["policy_revision"] == 1
    assert record["precision"] == profile.to_record()
    assert numerical_policy(profile.value) == policy
    assert NumericalPolicy(profile) == policy


def test_profile_cache_keys_and_json_are_distinct_and_deterministic():
    policies = [numerical_policy(profile) for profile in PrecisionProfile]
    assert [policy.cache_key for policy in policies] == [
        ("float64", 64, 64, 1, 1),
        ("float32-time64", 32, 64, 1, 1),
        ("float32", 32, 32, 1, 1),
        ("float16", 16, 16, 1, 1),
    ]
    records = [json.dumps(policy.to_record(), sort_keys=True) for policy in policies]
    assert len(set(records)) == 4
    assert records == [
        json.dumps(numerical_policy(profile).to_record(), sort_keys=True)
        for profile in PrecisionProfile
    ]
    assert numerical_policy("float32", time_precision="float64") == policies[1]
    assert numerical_policy("float32", time_precision="float32") == policies[2]


@pytest.mark.parametrize("profile", (
    PrecisionProfile.FLOAT32, PrecisionProfile.FLOAT32_TIME64,
))
def test_reduced_precision_capabilities_are_implemented_not_universal_certification(profile):
    policy = numerical_policy(profile)
    assert policy.preserves_existing_float64_settings is False
    assert policy.authorizes_execution is False
    assert set(policy.capabilities.to_record()) == {
        "analytical_propagation", "crossing", "root_finding",
        "adaptive_integration", "hazard", "mixed_clock_comparisons",
    }
    assert all(
        getattr(policy.capabilities, item.name) is NumericalStatus.IMPLEMENTED
        for item in fields(NumericalCapabilities)
    )
    assert require_supported_precision(profile) is profile
    assert policy.step_defaults.relative_tolerance == 64 * 2**-23
    assert policy.step_defaults.absolute_tolerance == 1e-10
    assert policy.step_defaults.event_tolerance == 8 * 2**-23


@pytest.mark.parametrize("profile", (
    PrecisionProfile.FLOAT32, PrecisionProfile.FLOAT32_TIME64,
))
def test_requirements_do_not_invent_tolerances_or_whole_runtime_widening_claims(profile):
    policy = numerical_policy(profile)
    assert policy.primitive_binding_scope == (
        "CONST", "PARAM", "NEG", "ADD", "SUB", "MUL", "DIV", "MAX",
    )
    assert policy.required_certifications == (
        "target-toolchain arithmetic and math-library validation",
        "device execution, memory capacity, and clock-horizon validation",
    )
    assert set(policy.to_record()) == {
        "schema_version", "policy_revision", "precision", "capabilities",
        "preserves_existing_float64_settings", "required_certifications",
        "primitive_binding_scope", "authorizes_execution",
    }


@pytest.mark.parametrize("precision", ("FLOAT64", "float", "float8", ""))
def test_noncanonical_profile_names_are_rejected(precision):
    with pytest.raises(ValueError, match="unknown precision"):
        numerical_policy(precision)


@pytest.mark.parametrize("precision", (None, True, 32, 64.0, object()))
def test_nonstring_profile_names_are_rejected(precision):
    with pytest.raises(TypeError, match="precision must be"):
        numerical_policy(precision)


@pytest.mark.parametrize("precision,time_precision", (
    ("float64", "float32"),
    ("float32-time64", "float32"),
    ("float32", "float16"),
))
def test_contradictory_clock_profiles_are_rejected(precision, time_precision):
    with pytest.raises(ValueError):
        numerical_policy(precision, time_precision=time_precision)


def test_record_constructor_does_not_allow_a_contradictory_policy():
    with pytest.raises(TypeError, match="PrecisionProfile"):
        NumericalPolicy("float32")
    with pytest.raises(TypeError):
        NumericalPolicy(PrecisionProfile.FLOAT32, policy_revision=2)
    with pytest.raises(TypeError):
        NumericalPolicy(PrecisionProfile.FLOAT32, authorizes_execution=True)

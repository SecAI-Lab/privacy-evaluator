import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

"""
Metric and Classifier based attacks by https://github.com/inspire-group/membership-inference-evaluation
Integrated version into TF-Privacy: 
    https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack
"""


def run_custom_attacks(attack_input):
    slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=True,
        by_percentiles=False,
        by_classification_correctness=True
    )

    """
    You can add more attacks from
    https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/data_structures.py#L172
    """
    metric_attacks = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.THRESHOLD_ENTROPY_ATTACK
    ]
    trained_attacks = [
        AttackType.LOGISTIC_REGRESSION,
        # AttackType.RANDOM_FOREST
    ]

    print("\nRunning Metric Attacks .....")
    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=metric_attacks)
    print(attacks_result.summary(by_slices=True))

    print("\nRunning Trained (Classifier) Attacks .....")
    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=trained_attacks)
    print(attacks_result.summary(by_slices=True))

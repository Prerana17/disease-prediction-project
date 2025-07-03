from disease_predictor.infer import infer_single_sample
import pytest

@pytest.mark.unit
@pytest.mark.functional
def test_inference_prediction():
    sample = {
        'Disease': 'Asthma',
        'Fever': 'Yes',
        'Cough': 'No',
        'Fatigue': 'Yes',
        'Difficulty Breathing': 'No',
        'Age': 30,
        'Gender': 'Male',
        'Blood Pressure': 'Normal',
        'Cholesterol Level': 'High'
    }
    pred, prob = infer_single_sample('Random_Forest', sample)
    assert isinstance(pred, int)
    assert len(prob) == 2

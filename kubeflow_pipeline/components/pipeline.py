from kfp import dsl
from kfp.components import load_component_from_file

@dsl.pipeline(
    name="ASR Pipeline",
    description="Pipeline for audio speech recognition"
)
def asr_pipeline(audio_input: str, model_path: str):
    preprocess_op = load_component_from_file('kubeflow_pipeline/components/preprocess/component.yaml')
    inference_op = load_component_from_file('kubeflow_pipeline/components/inference/component.yaml')
    postprocess_op = load_component_from_file('kubeflow_pipeline/components/postprocess/component.yaml')

    step1 = preprocess_op(input_path=audio_input)
    step2 = inference_op(audio_path=step1.outputs['output'], model_path=model_path)
    step3 = postprocess_op(input_txt=step2.outputs['output_txt'])

from .policy import PerturbCollect

# aic_model은 모듈 경로의 마지막 컴포넌트를 클래스명으로 찾음
# (예: policy:=data_gen_policy.policy.perturbcollect → 클래스 'perturbcollect' 탐색)
perturbcollect = PerturbCollect

# 文档智能下游任务代码
***

_be careful: 由于新版本的transformers的 layoutlmv3 的命名和 ms版本的命名有冲突，所以目前只能永达4.5.0版本，后续待fix_
* run_xfund.py # 官方版本xfund任务代码,可以验证layoutlmv3能否正常训练
* run_xfund_gp.py # global pointer 版本 xfund NER代码;
* run_t1_ner_gp.py # icdar 2023 task1 数据格式下的 global pointer ner 代码
* run_er_extraction_predict.py # icdar 2023 task1 格式下的 关系抽取 预测代码
* run_er_extraction_train.py # icdar 2023 task1 格式下的 关系抽取 的  训练代码

# 【小伙伴C】

VAL_RESULT_COMM 被 append 到 infer_cg。

  在 build_graph_atlas() 中：

  const ComputationGraph* cg = is_inference_graph(gid) ? infer_cg_ : train_cg_;

  VAL_RESULT_COMM 不是 inference graph（is_inference_graph 只返回 INF_MAIN_A/B 和 INF_EMA_A/B）。

  所以 cg = train_cg_。

  等等！VAL_RESULT_COMM 不是 inference graph！

  is_inference_graph(VAL_RESULT_COMM) 返回 false！

  所以 cg = train_cg_。

  但 VAL_RESULT_COMM 被 append 到 infer_cg_。

  在 build_graph_atlas() 中：

  if (!cg || cg->nodes(gid).empty()) continue;

  cg = train_cg_，train_cg_->nodes(VAL_RESULT_COMM) 是空的（因为 VAL_RESULT_COMM 被 append 到 infer_cg_）！

  所以 cg->nodes(gid).empty() 为 true！

  continue！slot.cg 保持为 nullptr！

  这就是问题！

  VAL_RESULT_COMM 被 append 到 infer_cg_，但 build_graph_atlas() 认为它是 train graph，所以查找 train_cg_->nodes(VAL_RES
  ULT_COMM)，这是空的！

  所以 VAL_RESULT_COMM 没有被填入 GraphAtlas！

  captured_idx = -1！

  resolve 返回 nullptr！

  vb_comm 是 nullptr！

  VAL_RESULT_ALLREDUCE 没有执行！

  这就是 root cause！

  VAL_RESULT_COMM 被错误地归类为 train graph，导致 build_graph_atlas() 没有为它创建 slot！
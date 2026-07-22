FROM oneframecontent/oneframe-ai-worker-v1:sha-7b36b02b3c40ca4439c9fad296cb70ac2da4cf09
WORKDIR /workspace/oneframe_job
COPY ai_worker_v1/training/pod_runner/runner.py /workspace/oneframe_job/runner.py
COPY ai_worker_v1/training/jobs/ball_v0_t3_pod.json /workspace/oneframe_job/job.json
COPY ai_worker_v1/training/ball_v0/t3_package/oneframe /workspace/oneframe_job/t3_package/oneframe
CMD ["python", "-u", "/workspace/oneframe_job/runner.py"]

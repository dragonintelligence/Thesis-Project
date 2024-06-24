import wandb, os
os.environ["WANDB_API_KEY"] ="53a0710ff054ea7108a9fc4bb93dff685e5eb957"
os.environ["WANDB_HTTP_TIMEOUT"] = "300"
wandb.login()
wandb.init(project="B", resume="allow", id="1")
for i in range(10):
    wandb.log({"a": i})
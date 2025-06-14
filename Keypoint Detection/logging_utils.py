# ==== Logging ====
log_file_path = "log_file.txt"
with open(log_file_path, "a") as log_file:
    current_time = datetime.datetime.now()
    log_message = f"{current_time}: Script executed successfully\n"
    log_file.write(log_message)
print("Script executed successfully")

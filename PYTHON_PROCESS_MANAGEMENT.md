# Managing Background Python Processes

This guide explains how to run `main.py` directly on the host VM in the background and manage it.

## 1. Running in the Background (Recommended)

When running over SSH, you should use `nohup` (No Hang UP) to ensure the process continues if your connection drops.

### Start the process
```bash
nohup python main.py > run.log 2>&1 &
```
*   `nohup`: Prevents the process from being killed when you disconnect.
*   `> run.log 2>&1`: Redirects both standard output and errors to `run.log`.
*   `&`: Puts the process in the background.

### Check if it's running
```bash
ps aux | grep main.py
```

---

## 2. Managing the Running Process

### Monitor the logs (Live)
```bash
tail -f run.log
```
Press `Ctrl+C` to stop viewing logs (the process keeps running).

### Bring to Foreground
If you started the process in your **current session** using `&` (or `Ctrl+Z` and `bg`), you can bring it back:
```bash
fg
```
*Note: This only works in the same terminal window where you started the process.*

### Pause and Background
If the process is running in your foreground and you want to move it to the background:
1. Press `Ctrl+Z` (Pauses the process).
2. Type `bg` and press Enter (Resumes it in the background).

---

## 3. Terminating the Process

### Graceful Termination
Find the Process ID (PID) and kill it:
```bash
# Find PID
ps aux | grep main.py

# Kill it (replace 1234 with the actual PID)
kill 1234
```

### Force Termination (If stuck)
```bash
kill -9 1234
```

### Kill by Name (Quickest)
```bash
pkill -f main.py
```

---

## 4. Best Practice: Using Screen/TMUX

For long-running pipelines, it is best to use a terminal multiplexer like `screen`. This allows you to disconnect and reconnect to the *same* interactive session.

### Start a session
```bash
screen -S vis-run
```

### Run your code
```bash
python main.py
```

### Detach (Leave it running)
Press `Ctrl+A` then `D`. You can now safely close your SSH connection.

### Reattach (Resume the session)
```bash
screen -r vis-run
```

### List all sessions
```bash
screen -ls
```

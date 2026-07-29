import os
from datetime import datetime

class Logger():
    def __init__(self, filename, is_debug, path='/home/weiw/Code/MM/KDMM/logs/'):
        # ===== Windows Logging Compatibility =====
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_log_path = os.path.join(repo_root, 'logs')
        if os.name == 'nt' and path == '/home/weiw/Code/MM/KDMM/logs/':
            path = local_log_path
        safe_filename = str(filename)
        if os.name == 'nt':
            for invalid_char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
                safe_filename = safe_filename.replace(invalid_char, '_')
        self.filename = safe_filename
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.log_ = not is_debug
    def logging(self, s):
        s = str(s)
        print(datetime.now().strftime('%Y-%m-%d %H:%M: '), s)
        if self.log_:
            with open(os.path.join(os.path.join(self.path, self.filename)), 'a+') as f_log:
                f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:  ')) + s + '\n')

import logging,os

heading_decorator = "\n-------------------------------------------\n"

def print_heading(heading):
	print(heading_decorator+heading+heading_decorator)

save_dir = "output/"
os.makedirs(os.path.dirname(save_dir), exist_ok=True)

debug_logfile = save_dir+"map_elites_debug.log"
logfile = save_dir+"map_elites.log"

# define a Handler which writes INFO messages or higher to the sys.stderr
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)-8s %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

# define a Handler which writes all messages to a debug log file
debug_file_handler = logging.FileHandler(debug_logfile,mode='a')
debug_file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M')
debug_file_handler.setFormatter(formatter)
logging.getLogger('').addHandler(debug_file_handler)

# define a Handler which writes all >=info messages to a log file
file_handler = logging.FileHandler(logfile,mode='a')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M')
file_handler.setFormatter(formatter)
logging.getLogger('').addHandler(file_handler)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def getLogger():
	return logger

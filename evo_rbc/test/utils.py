import logging

heading_decorator = "\n-------------------------------------------\n"

def print_heading(heading):
	print(heading_decorator+heading+heading_decorator)

def getLogger():
	debug_logfile = "ant_map_elites_debug.log"
	logfile = "ant_map_elites.log"

	# define a Handler which writes INFO messages or higher to the sys.stderr
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(levelname)-8s %(message)s')
	console_handler.setFormatter(formatter)
	logging.getLogger('').addHandler(console_handler)

	# define a Handler which writes all messages to a debug log file
	debug_file_handler = logging.FileHandler(debug_logfile,mode='w')
	debug_file_handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M')
	debug_file_handler.setFormatter(formatter)
	logging.getLogger('').addHandler(debug_file_handler)

	# define a Handler which writes all >=info messages to a log file
	file_handler = logging.FileHandler(logfile,mode='w')
	file_handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M')
	file_handler.setFormatter(formatter)
	logging.getLogger('').addHandler(file_handler)

	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	return logger
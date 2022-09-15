import wmh_utility
import wmh_pre
import wmh_mid_detection
import wmh_normalize
import vent_segment
import wmh_segment
import wmh_train
import wmh_create_dataset
import wmh_network
import sys

if __name__ == '__main__':
	desc_path = sys.argv[1]
	print('-------------------------------------', desc_path)
	desc, error1 = wmh_utility.load_description(desc_path)
	lpd_path = desc["networksPath"]+desc["number"]+"/"+desc["number"]+".lpd"
	model_path = desc["networksPath"]+desc["number"]+"/"+desc["number"]+"_best.npz"
	lp, arch, error2 = wmh_network.load_learning_process_description(lpd_path)
	if not (error1 or error2):
		index = 0
		train_cases = wmh_utility.load_cases(lp["trainingCases"])
		valid_cases = wmh_utility.load_cases(lp["validationCases"])
		test_cases = wmh_utility.load_cases(lp["testCases"])
		cases = train_cases + valid_cases + test_cases
		for c in cases:
			print("now for ", c)
			wmh_pre.run(desc["dataPath"], desc["atlasPath"], c, desc["files"])
			wmh_mid_detection.run(desc["dataPath"], c)
			wmh_normalize.run(desc["dataPath"], c)
		vent_segment.run_batch(desc["dataPath"], cases, desc["ventModelPath"], desc["ventArchPath"])
		wmh_create_dataset.run(desc["networksPath"],desc["number"], desc["dataPath"])
		wmh_train.run(desc["networksPath"],desc["number"])
		wmh_segment.run_batch(desc["dataPath"], test_cases, model_path, lpd_path, desc["outFileName"])
	else:
		print ("error with the provided description!")
		
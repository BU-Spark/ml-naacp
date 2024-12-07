import secret

from Mongo_Utils.mongo_funcs import get_collection

from global_state import global_instance

# Get all neighborhoods from the database for geocoding
def get_neighborhoods(client, org_key):
	db_prod = client[secret.db_name]
	try:
		neighborhood_collection = get_collection(db_prod, "neighborhood_data_" + org_key)
		tract_to_neighborhood = {}
		neighborhood_to_tracts = {}
		
		neighborhoods = neighborhood_collection.find()

		# Populate the dictionary with tract-to-neighborhood mappings
		for neighborhood in neighborhoods:
			neighborhood_name = neighborhood.get('value')
			tracts = neighborhood.get('tracts', [])
			
			for tract in tracts:
				tract_to_neighborhood[tract] = neighborhood_name

			if neighborhood_name not in neighborhood_to_tracts:
				neighborhood_to_tracts[neighborhood_name] = tracts

		return tract_to_neighborhood, neighborhood_to_tracts
	except Exception as error:
		print(f"[ERROR] Error getting neighborhoods from database: {error}")
		neigh_map = neigh_tract_dict
		tract_map = {}
		for neigh, tracts in neigh_tract_dict.items():
			for tract in tracts:
				tract_map[tract] = neigh
		return tract_map, neigh_map

def create_neighborhood(tract, neighborhood):
	neigh_map = global_instance.get_data("neigh_map")
	tract_map = global_instance.get_data("tract_map")
	
	if neighborhood not in neigh_map:
		neigh_map[neighborhood] = [tract]
	else:
		neigh_map[neighborhood].append(tract)
	tract_map[tract] = neighborhood
	global_instance.update_data("neigh_map", neigh_map)
	global_instance.update_data("tract_map", tract_map)

# Just in case, should not be used in production
neigh_tract_dict = {
	"Fenway" : ["010103", "010104", "010204", "010408", "010404", "010403", "981501", "010405", "010206", "010205"],
	"Downtown": ["030302", "070202", "070102", "030301", "070104", "070103", "070201"],
	"Beacon Hill": ["020200", "020302", "020101", "981700"],
	"Dorchester" : [
	"092400", "091400", "090300", "091800", "092300", "100601", "090901", 
	"100400", "090100", "091001","090200", "100200", "091700", "092200", "090700",
	"091500", "091300", "100300", "100100", "092000", "100500", "100800", "100603",
	"091200", "100700", "092101", "091900", "091600", "091100"
	],
	"Mattapan": ["100900", "101002", "101102", "981100", "101001","101101"],
	"Jamaica Plain": [
	"120103", "981800", "110105", "120600", "120700", "120301", "081200", "120105","081101",
	"981000", "120500", "120104", "120201", "110106", "081301", "120400"
	],
	"Roslindale": ["110502", "110104", "110501", "110401", "140106", "110301", "110607", "110403","110201"],
	"Roxbury": [
	"081500", "080500", "070801", "080100", "081800", "980300", "082000", "080601", "081700", "080300",
	"090600", "081400", "090400", "070901", "082100", "081900", "081302","080401"
	],
	"West End": ["020304", "020301", "020305"],
	"Longwood": ["010300", "081001"],
	"South Boston": ["061101", "060700", "060101", "061201", "061000", "060800", "981201", "060200", "061202", "060400", "061203", "060301", "060601", "060501"],
	"Back Bay": ["010702", "010701", "010802", "010801", "010500", "010600"],
	"Charlestown": ["040100", "040300", "040401", "040600", "040801", "040200"],
	"Allston": ["000604", "000804", "000703", "000704", "000806", "000101", "000807", "000701", "000805"],
	"Hyde Park": ["140107", "140201", "140105", "980700", "140300", "140202", "140400", "140102"],
	"East Boston": ["050500", "050600", "981502", "050101", "981300", "050901", "050300", "050700", "050400", "051000", "981600", "051200", "050200", "051101"],
	"South End": ["070301", "070302", "070502", "070501", "071101", "070600", "070700", "070902", "070802", "071201", "070402"],
	"West Roxbury": ["980900", "130406", "981900", "130404", "110601", "130300", "130402", "130200", "130101"],
	"South Boston Waterfront": ["981202", "060602", "060603", "061204", "060604"],
	"North End": ["030200", "030100", "030500", "030400"],
	"Cambridge": ["354300", "354200", "353102", "353600", "352300", "354100", "359400", "353300", "353700", "353200", 
	"354601", "355000", "354602", "354000", "354901", "354902", "353900", "354700", "352102", "354500", "354800", "352600", 
	"354400", "353101", "352900", "353000", "352101", "353800", "352500", "352400", "352700", "352200", "352800", 
    "365100", "361300"
  	],
	"Chelsea": ["160400", "160103", "160102", "160300", "160601", "160602", "160501", "160502", "160200"],
	

}
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the raw data
raw_data = [
    "59.166.0.0,1390,149.171.126.6,53,udp,CON,0.001055,132,164,31,29,0,0,dns,500473.9375,621800.9375,2,2,0,0,0,0,66,82,0,0,0,0,1421927414,1421927414,0.017,0.013,0,0,0,0,0,0,0,3,7,1,3,1,1,1,,0",
    "59.166.0.0,33661,149.171.126.9,1024,udp,CON,0.036133,528,304,31,29,0,0,-,87676.08594,50480.17188,4,4,0,0,0,0,132,76,0,0,9.89101,10.682733,1421927414,1421927414,7.005,7.564333,0,0,0,0,0,0,0,0,2,4,2,3,1,1,2,,0",
    "59.166.0.6,1464,149.171.126.7,53,udp,CON,0.001119,146,178,31,29,0,0,dns,521894.5313,636282.375,2,2,0,0,0,0,73,89,0,0,0,0,1421927414,1421927414,0.017,0.013,0,0,0,0,0,0,0,0,12,8,1,2,2,1,1,,0",
    "59.166.0.5,3593,149.171.126.5,53,udp,CON,0.001209,132,164,31,29,0,0,dns,436724.5625,542597.1875,2,2,0,0,0,0,66,82,0,0,0,0,1421927414,1421927414,0.043,0.014,0,0,0,0,0,0,0,0,6,9,1,1,1,1,1,,0",
    "59.166.0.3,49664,149.171.126.0,53,udp,CON,0.001169,146,178,31,29,0,0,dns,499572.25,609067.5625,2,2,0,0,0,0,73,89,0,0,0,0,1421927414,1421927414,0.005,0.003,0,0,0,0,0,0,0,0,7,9,1,1,1,1,1,,0",
    "59.166.0.0,32119,149.171.126.9,111,udp,CON,0.078339,568,312,31,29,0,0,-,43503.23438,23896.14258,4,4,0,0,0,0,142,78,0,0,29.682221,34.37034,1421927414,1421927414,21.003,24.315,0,0,0,0,0,0,0,0,2,4,2,3,1,1,2,,0",
    "59.166.0.6,2142,149.171.126.4,53,udp,CON,0.001134,132,164,31,29,0,0,dns,465608.4688,578483.25,2,2,0,0,0,0,66,82,0,0,0,0,1421927414,1421927414,0.017,0.013,0,0,0,0,0,0,0,0,12,7,1,2,2,1,1,,0"
]

# Convert the raw data into a DataFrame
columns = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
           "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
           "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
           "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
           "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
           "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"]

raw_df = pd.DataFrame([row.split(",") for row in raw_data], columns=columns)

# Load the Random Forest model from the .pkl file
rf_model = pickle.load(open("H:\CyberProject\Random Forest_model.pkl", "rb"))

# Preprocess the raw data (you need to apply the same preprocessing steps as in your training pipeline)
# Assume you have functions clean_data, apply_log1p, standardize, ohencoding defined

# Make predictions using the loaded model
predictions = rf_model.predict(raw_df.drop(["attack_cat", "Label"], axis=1))

# Output the predictions
print("Predictions:")
print(predictions)

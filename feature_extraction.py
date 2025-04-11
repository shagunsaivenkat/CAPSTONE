import csv
import os
import sys
import struct
import socket
from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP, ICMP

def ip_to_int(ip):
    """ Convert an IP address string to an integer """
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def extract_features_from_pcap(file_path, output_path):
    flows = {}
    try:
        with PcapReader(file_path) as packets:
            for pkt in packets:
                if IP in pkt:
                    src_ip = ip_to_int(pkt[IP].src)  # Convert IP to int (Move outside)
                    dst_ip = ip_to_int(pkt[IP].dst)  # Convert IP to int (Move outside)
                    protocol = pkt[IP].proto
                    src_port, dst_port, payload_size, flags = 0, 0, 0, 0  # Default values

                    if TCP in pkt:
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                        payload_size = len(pkt[TCP].payload)
                        flags = int(pkt[TCP].flags)
                    elif UDP in pkt:
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport
                        payload_size = len(pkt[UDP].payload)
                    elif ICMP in pkt:
                        payload_size = len(pkt[ICMP].payload)

                    flow_id = (src_ip, src_port, dst_ip, dst_port, protocol)
                    if flow_id in flows:
                        flows[flow_id]['packet_count'] += 1
                        flows[flow_id]['total_bytes'] += payload_size
                    else:
                        flows[flow_id] = {
                            'src_ip': src_ip,
                            'dst_ip': dst_ip,
                            'src_port': src_port,
                            'dst_port': dst_port,
                            'protocol': protocol,
                            'packet_count': 1,
                            'total_bytes': payload_size,
                            'tcp_flags': flags
                        }

        # Write to CSV
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'packet_count', 'total_bytes', 'tcp_flags'])
            writer.writeheader()
            for data in flows.values():
                writer.writerow(data)

        print(f"Feature extraction complete. Saved to {output_path}")

    except Exception as e:
        print(f"Error in feature extraction: {e}", file=sys.stderr)
        raise  

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Feature_Extraction.py <input_pcap> <output_csv>", file=sys.stderr)
        sys.exit(1)
    input_pcap = sys.argv[1]
    output_csv = sys.argv[2]
    extract_features_from_pcap(input_pcap, output_csv)

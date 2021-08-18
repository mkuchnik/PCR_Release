#include <iostream>
#include <fstream>

#include "pcr.h"
#include "proto/LightMLRecords.pb.h"

int main(int argc, char* argv[]) {
  std::cout << "Hello world" << std::endl;

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
    return -1;
  }
  std::fstream input(argv[1], std::ios::in | std::ios::binary);

  if (!input) {
    std::cerr << "File not found" << std::endl;
  }
 
  // TODO
  std::vector<size_t> record_offsets = {66, 8683, 3394, 4418, 4446, 13414, 6030, 609, 2509, 2482, 6051};

  LightMLRecords::MetadataRecord metarecord;
  std::vector<LightMLRecords::ScanGroup> scangroups;

  std::size_t last_offset = 0;
  for (std::size_t i = 0; i < record_offsets.size(); i++) {
    std::size_t offset = record_offsets.at(i);
    std::cout << "Reading up to " << offset << std::endl;
    std::size_t size = offset - last_offset;
    std::vector<char> buffer(size);
    input.read(&buffer[0], buffer.size());
    std::string content(buffer.begin(), buffer.end());
    if (i == 0) {
      int ret = metarecord.ParseFromString(content);
      if (!ret) {
        std::cerr << "Failed to parse" << std::endl;
      }
    } else {
      LightMLRecords::ScanGroup scangroup;
      int ret = scangroup.ParseFromString(content);
      if (!ret) {
        std::cerr << "Failed to parse" << std::endl;
      }
      scangroups.push_back(scangroup);
    }
  }
}


#include "pcr.h"

#define _GNU_SOURCE
#include <fcntl.h>
#include <malloc.h>
#include <unistd.h>
#include <errno.h>

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <exception>

// TODO disable for production
//#define PCR_DEBUG

#ifdef PCR_DEBUG
#include <chrono>
#endif

#include "proto/LightMLRecords.pb.h"

std::pair< std::vector< std::string >, std::vector<int> > load_PCR(
    std::string filename,
    std::vector<int> record_offsets,
    int n_scans
    ) {
  assert(n_scans > 0);
  assert(n_scans < record_offsets.size());

  int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
  if (fd == -1) {
    std::stringstream err_msg;
    err_msg << "Cannot open file: ";
    err_msg << filename;
    std::cerr << err_msg.str() << std::endl;
    throw std::runtime_error(err_msg.str());
  }

  // TODO posix_fadvise don't need
  //std::fstream input(filename, std::ios::in | std::ios::binary);
  //if (!input) {
  //  std::cerr << "File not found" << std::endl;
  //}

  // We need to do a single large read
  std::size_t total_bytes_read = record_offsets.at(n_scans);
  const size_t alignment = 4096;
  std::size_t total_chunks_needed = total_bytes_read / alignment;
  if (total_bytes_read % alignment) {
    total_chunks_needed += 1;
  }
  std::size_t total_bytes_needed = total_chunks_needed * alignment;

  char* buf = (char*) memalign(alignment, total_bytes_needed);
  if (buf == NULL) {
    std::stringstream err_msg;
    err_msg << "Cannot allocate memory for filename: ";
    err_msg << filename;
    err_msg << ". Errno: ";
    err_msg << strerror(errno);
    std::cerr << err_msg.str() << std::endl;
    throw std::runtime_error(err_msg.str());
  }

  //std::string buffer(total_bytes_read, 0);
  #ifdef PCR_DEBUG
  auto start_time = std::chrono::high_resolution_clock::now();
  int numRead = read(fd, buf, total_bytes_needed);
  if (numRead == -1) {
    std::stringstream err_msg;
    err_msg << "Cannot read file: ";
    err_msg << filename;
    err_msg << ". Errno: ";
    err_msg << strerror(errno);
    std::cerr << err_msg.str() << std::endl;
    throw std::runtime_error(err_msg.str());
  }
  auto stop_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
  std::cout << "duration read " << " = " << duration.count() << std::endl;
  #else
  int numRead = read(fd, buf, total_bytes_needed);
  if (numRead == -1) {
    std::stringstream err_msg;
    err_msg << "Cannot read file: ";
    err_msg << filename;
    err_msg << ". Errno: ";
    err_msg << strerror(errno);
    std::cerr << err_msg.str() << std::endl;
    throw std::runtime_error(err_msg.str());
  }
  #endif

  int ret = close(fd);
  if (ret == -1) {
    std::stringstream err_msg;
    err_msg << "Cannot close file: ";
    err_msg << filename;
    err_msg << ". Errno: ";
    err_msg << strerror(errno);
    std::cerr << err_msg.str() << std::endl;
    // Non-fatal: keep going
  }


  LightMLRecords::MetadataRecord metarecord;
  std::vector<LightMLRecords::ScanGroup> scangroups;
  std::size_t last_offset = 0;  
  std::size_t expected_bytes = 0; // We use the number of bytes seen as a proxy for image size
  for (std::size_t i = 0; i <= n_scans; i++) {
    std::size_t offset = record_offsets.at(i);
    std::size_t size = offset - last_offset;
    std::string content = std::string(buf + last_offset, size);
    #ifdef PCR_DEBUG
    std::cout << "content " << last_offset << " " << offset <<  std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    #endif
    if (i == 0) {
      // First record is metadata
      int ret = metarecord.ParseFromString(content);
      if (!ret) {
        std::cerr << "Failed to parse" << std::endl;
      }
    } else {
      expected_bytes += offset;
      LightMLRecords::ScanGroup scangroup;
      int ret = scangroup.ParseFromString(content);
      if (!ret) {
        std::cerr << "Failed to parse" << std::endl;
      }
      scangroups.push_back(scangroup);
    }
    #ifdef PCR_DEBUG
    stop_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    std::cout << "duration decode " << i << " = " << duration.count() << std::endl;
    #endif
    last_offset = offset;
  }

  // Free the memory
  free(buf);

  // Normalize
  expected_bytes = expected_bytes / n_scans + 1;

  const size_t N_images = metarecord.labels().size();

  std::vector<int> Y;
  Y.reserve(N_images);
  for (auto& label: metarecord.labels()) {
    Y.push_back(label);
  }

  // Preallocate
  std::vector< std::string > all_image_bytes(N_images);


  for (auto& scangroup: scangroups) {
    // This contains N==len(Y) partial images
    auto grouped_partial = scangroup.image_bytes();
    assert(grouped_partial.size() == N_images);
    for (std::size_t i = 0; i < grouped_partial.size(); i++) {
      std::vector<char> partial_image(grouped_partial[i].size());
      for (std::size_t j = 0; j < grouped_partial[i].size(); j++) {
        partial_image[j] = grouped_partial[i][j];
      }
      std::size_t n_new_bytes = partial_image.size();
      all_image_bytes[i].reserve(all_image_bytes[i].size() + n_new_bytes);
      all_image_bytes[i].insert(
          all_image_bytes[i].end(),
          partial_image.begin(),
          partial_image.end()
     );
    }
  }

  const unsigned char end_of_image[2] = {0xFF, 0xD9};
  for (auto& image_bytes: all_image_bytes) {
    std::size_t str_len = image_bytes.length();
    if ((image_bytes[str_len-2] != end_of_image[0]) ||
        (image_bytes[str_len-1] != end_of_image[1])) {
       image_bytes.append((const char*) end_of_image, 2);
    }
  }

  auto pair = std::make_pair(all_image_bytes, Y);
  return pair;
}

#ifndef PCR_H
#define PCR_H

#include <string>
#include <vector>

/**
 * Inverse function of create_progressive_compressed_tf_record
 * @param A list of data corresponding to [metadata, scan_0, scan_1,...] for
 * potentially many images
 * @param n_scans Only n_scans are required. This should match records
 * @return The bytes of JPEG images and the labels as tuple (X, Y)
 */
std::pair< std::vector< std::string >, std::vector<int> > load_PCR(
    std::string filename,
    std::vector<int> record_offsets,
    int n_scans
);

#endif // PCR_H
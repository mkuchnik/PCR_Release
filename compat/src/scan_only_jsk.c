/*
 * jsk.c
 *
 * Copyright (C) 2013, Frederic Kayser.
 * From:
 * https://encode.su/threads/1800-JSK-JPEG-Scan-Killer-progressive-JPEG-explained-in-slowmo
 *
 */


#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef WIN32
#  include <io.h>
#endif

#define errorxt(msg)  (fprintf(stderr, "%s\n", msg), exit(1))
#define jsk_version "0.1 (24 Sept. 2013)"

/*
 * JPEG markers consist of one or more 0xFF bytes, followed by a marker
 * code (which is not 0xFF).
 */

#define M_PAD   0x00   /* Padding byte used when 0xFF is found in compressed data */
#define M_RST0  0xD0   /* Restart Markers 0-7, related to DRI */
#define M_RST1  0xD1
#define M_RST2  0xD2
#define M_RST3  0xD3
#define M_RST4  0xD4
#define M_RST5  0xD5
#define M_RST6  0xD6
#define M_RST7  0xD7
#define M_SOI   0xD8   /* Start of Image */
#define M_EOI   0xD9   /* End of Image */
#define M_SOS   0xDA   /* Start of Scan (size is not known) */
#define M_DRI   0xDD   /* Define Restart Interval */

#define local static   /* for local function definitions */

/* These are global variables, this is baaaad !!! */
uint8_t *jpg_buff;
size_t size, cur_pos;

/* Read a 16 bits value from the buffer */
static uint16_t read_two (void)
{
  if ((cur_pos+1) < size) {
    uint16_t beammeup;
    beammeup = jpg_buff[cur_pos++] << 8;
    beammeup += jpg_buff[cur_pos++];
    return beammeup;
  }
  else
    errorxt("Premature EOF in JPEG file");
}

/* Read an 8 bits value from the buffer */
static uint8_t read_one (void)
{
  if (cur_pos < size)
    return (uint8_t)(jpg_buff[cur_pos++]);
  else
    errorxt("Premature EOF in JPEG file");
}

/* Find the next JPEG marker */
static uint8_t next_marker (void)
{
  uint8_t c;

  /* Find 0xFF byte (marker head). */
  
  c = read_one();
  while (c != 0xFF) {
    c = read_one();
  }
  
  /* Get marker code, swallowing any duplicate FF bytes.
   * (extra FFs are legal as pad bytes)
   */
  do {
    c = read_one();
  } while (c == 0xFF);

  return c;
}

/* Skip over a marker of known size */
static void skip_marker (void)
{
  uint16_t length;

  /* Read the marker length */
  length = read_two();

  /* Length includes itself, must be at least 2 */
  if (length < 2)
    errorxt("Erroneous JPEG marker length");
  length -= 2;

  /* Skip length bytes */
  if (cur_pos+length < size)
    cur_pos += length;
  else
    errorxt("Erroneous JPEG marker length");
}

static void skip_SOS (void)
{
  uint8_t c;

  /* Since the scan size is not known in advance it is parsed until we find a new marker */

  /* Find 0xFF bytes inside current scan, these could be internal markers */

  for (;;) {
    c = read_one();
    while (c != 0xFF) {
      c = read_one();
    }

    do {
      c = read_one();
    } while (c == 0xFF);

    switch (c) {
      case M_PAD:  break;  /* 0xFF was part of the compressed stream and was 0x00 padded*/
      case M_RST0:  c = 0 ; break;  /* Swallow eventual Restart Markers */
      case M_RST1:  c = 0 ; break;
      case M_RST2:  c = 0 ; break;
      case M_RST3:  c = 0 ; break;
      case M_RST4:  c = 0 ; break;
      case M_RST5:  c = 0 ; break;
      case M_RST6:  c = 0 ; break;
      case M_RST7:  c = 0 ; break;
      default:  break;
    }
    if (c!=0) {                  /* The end of the current scan has been reached */
      cur_pos -= 2;              /* Rollback for next_marker */
      break;                     /* Leave the loop */
    }
  }
}

void usage(FILE *fpMsg)
{
  fprintf(fpMsg, "JSK, version %s by Frederic Kayser.\n", jsk_version);
  fprintf(fpMsg,
    "Splits a JPEG file in multiple scan_xxx.jpg files.\n"
    "Usage: jsk [-h] file.jpg\n"
    "Options:\n"
    "    -h  display this help message\n");
}

int main(int argc, char *argv[])
{
  uint32_t i = 1;
  struct stat sbuf;  
  char *fname1;
  uint8_t marker;
  size_t data_io_size;
  char out_name[] = "scan_000.jpg";
  // unused
  // uint8_t ffd9[2] = {0xFF, 0xD9};
  FILE *fp1;

  while (argc > 1 && argv[1][0] == '-') {
    switch (argv[1][i]) {
      case '\0':
        --argc;
        ++argv;
        i = 1;
        break;
      case 'h':
        usage(stdout);
        exit (0);

      default:
        fprintf(stderr, "Unknown option %c\n\n", argv[1][i]);
        usage(stderr);
        exit (1);
    }
  }

  if (argc != 2) {
    usage(stderr);
    exit (1);
  }
  else {                   /* Open the input file and load it in memory */
    fname1 = argv[1];
    
    if (lstat(fname1, &sbuf)) {
      fprintf(stderr, "%s: does not exist!\n", fname1);
      exit (1);
    }
    if (S_ISDIR(sbuf.st_mode)) {
      fprintf(stderr, "%s: is a folder - ignored\n", fname1);
      exit (1);
    }
    size = sbuf.st_size;
    
    if ((fp1 = fopen(fname1, "rb")) == NULL) {
      fprintf(stderr, "%s: could not open file!\n", fname1);
      exit (1);
    }

    jpg_buff = malloc(size);
    if (jpg_buff == NULL) {
      fprintf(stderr, "Malloc failed miserably!\n");
      exit (1);
    }

    data_io_size = fread(jpg_buff, 1, size, fp1);
    if (data_io_size != (size_t)(size)) {
      fprintf(stderr, "%s: EOF while reading input!\n", fname1);
      exit (1);
    }
    fclose(fp1);


    /* Check that the file is effectively a JPEG */

    if (read_two() != 0xFFD8)
      errorxt("Input is not a JPEG file");
  
    /* Scan markers until EOI and treat SOS */
    do {
      marker = next_marker();
      if (marker == M_SOS) {
        skip_SOS(); 		/* Jump to the end of the scan */

        /* Increment the scan counter in the output file name */
        if (out_name[7]!='9')
          ++out_name[7];
        else {
          out_name[7]='0';
          if (out_name[6]!='9')
            ++out_name[6];
          else {
            out_name[6]='0';
            ++out_name[5];      /* 999 scans ought to be enough for anybody */
          }
        }

        /* Write everything from the begining of the file to the end of the current scan */
        //fp1 = fopen(out_name, "wb");
        //if (fp1 == NULL) {
        //  fprintf(stderr, "Could not open output file: %s.\n", out_name);
        //  exit (1);
        //}

        fprintf(stdout,
            "%zu\n",
            cur_pos);
        // data_io_size = fwrite(jpg_buff, 1, cur_pos, fp1);
        //if (data_io_size != cur_pos) {
        //  errorxt("Could not write entire output file!");
        //}

        /* Add an End of Image marker at the end of the file */

        // TODO we disable image writes
        // data_io_size = fwrite(ffd9, 1, 2, fp1);
        //if (data_io_size != 2) {
        //  errorxt("Could not add End Of Image to the output file!");
        //}
        //fclose(fp1);
      }
      else if (marker != M_EOI) {  /* Anything else is skipped */
        skip_marker();
      }
    } while (marker != M_EOI);     /* End of Image: we are done */
  }
  return 0;
}

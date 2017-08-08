# -*- coding: utf-8 -*-
# Reads RDF-format radar data files
# Using Python 2
# Copied and modified from [...]/FDL/Read_RDF/Read_RDF_v3.ipynb
# By Sean Marshall
# Last modified on Monday, July 17, 2017
#%matplotlib inline
import numpy
import matplotlib.pyplot as plt
import struct # file:///usr/share/doc/python-doc/html/library/struct.html

# The tests in this function may fail if it encounters chunks that are different
#   from those in the RDFs that I used during testing.
def is_text(chunk):
    if (len(chunk) > 16) and (len(chunk) < 192):
        # The longest chunk of text that I saw in the test data was 114
        # But I'll use a cutoff of 192, to be safe
        if (min(chunk[1:]) >= 32) and (max(chunk[1:]) <= 128):
            return True
    return False

null_value = -1.0e15
def read_rdf(file_path):
    """Reads a file of the radar data format (RDF), returning dict"""
    inp = numpy.fromfile(file_path, dtype=numpy.uint8)
    # Find where inp has bytes with value 0x0a (decimal 10)
    # Since those may indicate separations between lines of text
    ind010 = numpy.where(inp == 10)[0] # Indices separating possible chunks of text
    # Now process those chunks
    chunk = str(bytearray(inp[0:ind010[0]]))
    chunk_spl = chunk.split()
    temp = chunk.find(chunk_spl[2])
    outp = [('source_file', file_path), (chunk_spl[1], chunk[temp:])]
    text = [ chunk ] # Also keep a separate list with all text
    # First chunk should be text (NOT binary data)
    # Since some numeric tags include comments that may be needed later
    height = 0 # Number of delay rows, or number of CW spectra
    width = 0 # Number of frequency bins per row
    bytes_per_point = 0 # Size of each data value
    ntags = 0 # Number of tags that are stored as binary data (NOT text)
    tag_names = [ ] # Will store names of tags
    tag_indices = [ ] # Will store indices of tags
    ndata = -32768 # Number of non-tag data points in each image or spectrum
    #data_start = [ ] # Will store the starting positions of each block of data
    ici = 0 # Counter for all chunks
    icd = 0 # Counter for data (non-text) chunks that have been parsed
    for ici in range(ind010.size - 1):
        # The final chunk should be just a tab, so skip it
        chunk = bytearray(inp[ind010[ici]:ind010[ici+1]])
        if is_text(chunk[1:]):
            chunk = str(chunk[1:]) # Skip the first character (newline)
            text.append(chunk)
            #print chunk # For debugging
            chunk_spl = chunk.split()
            if chunk[0] == 'c' or chunk[0] == 's':
                temp = chunk.find(chunk_spl[2])
                outp.append((chunk_spl[1], chunk[temp:]))
            elif chunk[0] == 'v':
                # Vector of values, e.g. "v       stat[5] 14 12 13 14 0"
                #print chunk # For debugging
                ind_temp = [chunk.find(chunk_spl[1]), chunk.find('['), chunk.find(']')]
                #print chunk[ind_temp[0]:ind_temp[1]], chunk[ind_temp[2]+1:]
                #outp.append((chunk[ind_temp[0]:ind_temp[1]], chunk[ind_temp[2]+1:]))
                temp = chunk[ind_temp[2]+1:].split()
                v_values = [ ]
                icv = 0 # Counter for vector values
                for icv in range(len(temp)):
                    v_values.append(float(temp[icv]))
                #print v_values
                if int(chunk[ind_temp[1]+1:ind_temp[2]]) != len(v_values):
                    print "Warning: unexpected length for vector " + \
                        chunk[ind_temp[0]:ind_temp[1]]
                outp.append((chunk[ind_temp[0]:ind_temp[1]], numpy.array(v_values)))
            elif chunk[0] == 't':
                # Name of numeric tag
                # Tag values are stored as binary data after each "row" of data
                tag_names.append(chunk_spl[1])
                tag_indices.append(int(chunk_spl[2]))
                #print "  New tag: " + chunk_spl[1] # For debugging
            elif chunk[0] == 'd':
                outp.append((chunk_spl[1], float(chunk_spl[2])))
            elif chunk[0] == 'i':
                outp.append((chunk_spl[1], int(chunk_spl[2])))
                if chunk_spl[1].lower() == "height":
                    height = int(chunk_spl[2])
                elif chunk_spl[1].lower() == "width":
                    width = int(chunk_spl[2])
                elif chunk_spl[1].lower() == "size":
                    bytes_per_point = int(chunk_spl[2])
                elif chunk_spl[1].lower() == "ntags":
                    ntags = int(chunk_spl[2])
                elif chunk_spl[1].lower() == "ndata":
                    ndata = int(chunk_spl[2])
            elif chunk[0] == 'f':
                outp.append((chunk_spl[1], float(chunk_spl[2])))
            elif chunk[0] == '#':
                pass
            elif chunk[0] == ' ':
                pass
            else:
                print "Error in chunk:"
                print chunk
                raise ValueError("Unrecognized tag format")
        else:
            icd += 1
            #print icd, ind010[ici], ind010[ici+1]
            #print inp[ind010[ici]:ind010[ici]+4]
            if icd == 2:
                row_start = ind010[ici] + 1
                if ntags > 0:
                    # Need to handle initial tag separately
                    # Since it is not preceded by 0x0a
                    if len(tag_names) > 0:
                        raise ValueError("Name of tag 0 was set before data block")
                    if height > 0 and width > 0:
                        ind_tag0 = row_start + height*width*bytes_per_point
                        ind_tag1 = ind010[numpy.min(numpy.where(ind010 > ind_tag0)[0])]
                        #print "Addresses for start and end of tag 0:", ind_tag0, ind_tag1
                        chunk = bytearray(inp[ind_tag0:ind_tag1])
                        if is_text(chunk):
                            chunk = str(chunk) # No leading newline to skip
                            text.append(chunk)
                            #print chunk # For debugging
                            chunk_spl = chunk.split()
                            if chunk[0] == 't':
                                tag_names.append(chunk_spl[1])
                                tag_indices.append(int(chunk_spl[2]))
                                #print "  New tag: " + chunk_spl[1] # For debugging
                        else:
                            raise ValueError("No text at expected address of first tag")
                    else:
                        raise ValueError("Data block began before height and width were set")
            #else:
            #    if len(chunk) > 2:
            #        data_end = ind010[ici] + 1
    if height <= 0:
        raise ValueError("Problem with height")
    if width <= 0:
        raise ValueError("Problem with width")
    if bytes_per_point <= 0:
        raise ValueError("Problem with bytes_per_point")
    if ndata < 0:
        #print "Warning: ndata had been set to", ndata
        ndata = width - ntags
    if ndata + ntags != width:
        raise ValueError("Problem with ndata")
    #data_start.append(data_start[0] + width*bytes_per_point)
    #data_start = numpy.array(data_start)
    row_start = row_start + numpy.arange(height)*width*bytes_per_point
    row_end = row_start + ndata*bytes_per_point
    #data_end = data_start + height*ndata*bytes_per_point
    #print data_start, data_end, data_end - data_start
    #print inp[data_start:data_start+4]
    #print inp[data_end-4:data_end]
    #outp.append(('data', inp[data_start:data_end]))
    #N_pts = height*width
    data = null_value*numpy.ones((height, ndata))
    tag_values = null_value*numpy.ones((height, ntags))
    #      Initialize these arrays by filling them with an invalid value
    ici = 0 # Counter for rows of D-D images, or for which spectrum in a series
    #ick = data_start - bytes_per_point # Counter for address of inp
    for ici in range(height):
        icj = 0 # Counter for frequency (in D-D pixels or frequency bins), then for tags
        ick = row_start[ici] - bytes_per_point
        for icj in range(width):
            ick += bytes_per_point
            if ick < row_end[ici]:
                # If it's still in the data block
                data[ici,icj] = struct.unpack('>f', inp[ick:ick+bytes_per_point])[0]
                # Previous line assumes float values are stored in big-endian format!
            else:
                # If it's in the tags
                tag_values[ici,icj-ndata] = struct.unpack('>f', inp[ick:ick+bytes_per_point])[0]
    #print tag_names # For debugging
    #print tag_indices
    #ainfo(tag_values)
    #print tag_values
    icj = 0 # Counter for tags
    if ntags > 0:
        for icj in range(ntags):
            #print tag_names[icj], tag_indices[icj], tag_values[:,tag_indices[icj]]
            outp.append((tag_names[icj], tag_values[:,tag_indices[icj]]))
    outp.append(('data', data))
    outp.append(('text', text))
    return dict(outp)

font_size = 16
def plot_rdf_img(inp):
    """Given a dictionary from read_rdf, displays data"""
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
    # http://matplotlib.org/examples/color/colormaps_reference.html
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(inp['data'], cmap='gray')
    plt.title(inp['source_file'].split('/')[-1], fontsize=font_size+8)
    plt.xlabel("Doppler column", fontsize=font_size+4)
    plt.ylabel("Delay row", fontsize=font_size+4)
    cbar = plt.colorbar()
    cbar.set_label("SNR", fontsize=font_size+4)
    cbar.ax.tick_params(labelsize=font_size)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(font_size)
    plt.show()

def plot_rdf_cw(inp):
    """Given a dictionary from read_rdf, displays CW spectra"""
    fig = plt.figure(figsize=(12, 10))
    if "nchan" in inp.viewkeys():
        if inp['nchan'] > 1:
            plt.plot(inp['data'][0,:], ':.b')
            plt.plot(inp['data'][1,:], ':.r')
        else:
            plt.plot(inp['data'], ':.b')
    else:
        plt.plot(inp['data'], ':.b')
    plt.title(inp['source_file'].split('/')[-1], fontsize=font_size+8)
    plt.xlabel("Doppler column", fontsize=font_size)
    plt.ylabel("Signal-to-noise ratio", fontsize=font_size)
    plt.grid(True)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(font_size)
    plt.show()

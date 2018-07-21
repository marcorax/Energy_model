
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <Davisloading.hpp>

//The class is supposed to deal with AEDAT 3.1 files.
// To know more and to understand what is going on in the lines underneat 
// please refer to page https://inivation.com/support/software/fileformat/#aedat-31


void skip_header(std::ifstream & file){
    std::string first_letter = "#";
    std::string line;
    std::getline(file,line);
    int header_found = 0;
    while (first_letter=="#")
    {
        if(line=="#!END-HEADER\r"){
            std::cout<<"Header successfully skipped"<<std::endl;
            header_found=1;
            break;
        }
        else{
                std::getline(file,line);
                first_letter = line[0];
        }
    }
    if(header_found==0)
        std::cout<<"Header end not found"<<std::endl;
}

int read_frames(std::ifstream & file, int XDIM, int YDIM, std::vector <cv::Mat> & frames,
    std::vector <unsigned int> & start_ts,
    std::vector <unsigned int> & end_ts){
    unsigned short eventtype, eventsource;
    unsigned int eventsize, eventoffset, eventtsoverflow,
    eventcapacity, eventnumber, eventvalid, next_read;
    next_read = eventcapacity*eventsize;
    unsigned char * event_head = new unsigned char [28];
    file.read((char*) event_head, 28);
    if(file.tellg()==EOF){   
        return -1;
    }
    //The machine is supposed to be little endian.
    eventtype = (unsigned short) ((event_head[1]) << 8 | (event_head[0]));
    eventsource =  (unsigned short) ((event_head[3]) << 8 | (event_head[2]));
    eventsize =  (unsigned int) ((event_head[7]) << 24 | (event_head[6]<<16)|
                                 (event_head[5]) << 8  | (event_head[4]));
    eventoffset =  (unsigned int) ((event_head[11]) << 24 | (event_head[10]<<16)|
                                   (event_head[9])  << 8  | (event_head[8]));
    eventtsoverflow = (unsigned int) ((event_head[15]) << 24 | (event_head[14]<<16)|
                                      (event_head[13]) << 8  | (event_head[12]));
    eventcapacity = (unsigned int) ((event_head[19]) << 24 | (event_head[18]<<16)|
                                    (event_head[17]) << 8  | (event_head[16]));
    eventnumber = (unsigned int) ((event_head[23]) << 24 | (event_head[22]<<16)|
                                  (event_head[21]) << 8  | (event_head[20]));
    eventvalid = (unsigned int) ((event_head[27]) << 24 | (event_head[26]<<16)|
                                 (event_head[25]) << 8  | (event_head[24]));
    
    unsigned char * data = new unsigned char [eventsize];

    if(eventtype == 2){
    unsigned int pixelcounter=36;
    int posx = 0;
    int posy = 0;

        for(int i=0; i<eventcapacity;i++){
            file.read((char*) data, eventsize);
            posx=0;
            posy=0;
            cv::Mat tmpframe(YDIM, XDIM, CV_16UC1);
            while(pixelcounter<eventsize){
                tmpframe.at<ushort>(posy,posx)=((unsigned int) ((data[(pixelcounter)+1]) << 8 |
                                                     (data[(pixelcounter)])));
                posx++;
                if(posx==XDIM){
                    posx=0;
                    posy++;
                }
                pixelcounter += 2;
            }
            frames.push_back(tmpframe);
            start_ts.push_back((unsigned int) ((data[7]) << 24 | (data[6]<<16)|
                                               (data[5]) << 8  | (data[4])));
            end_ts.push_back((unsigned int) ((data[11]) << 24 | (data[10]<<16)|
                                              (data[9]) << 8  | (data[8])));

        }
    return 0;
    }

    if(eventtype==0){ //special event, the class is not developed to deal with this type, here for debugging only.
    std::vector <unsigned int> spec_ts;
    std::vector <unsigned short> spec_type;
        for(int i=0; i<eventcapacity;i++){
            file.read((char*) data, eventsize);
            spec_ts.push_back((unsigned int) ((data[7]) << 24 | (data[6]<<16)|
                                              (data[5]) << 8  | (data[4])));
            spec_type.push_back((unsigned short)(data[0]));
        }
    
    return 0;

    }
    
}

int read_events(std::ifstream & file, std::vector <unsigned short> & polarity,
    std::vector <unsigned int> & timestamp,
    std::vector <unsigned int> & x_addr,
    std::vector <unsigned int> & y_addr){

    unsigned short eventtype, eventsource;
    unsigned int eventsize, eventoffset, eventtsoverflow,
    eventcapacity, eventnumber, eventvalid, next_read;
    next_read = eventcapacity*eventsize;
    unsigned char * event_head = new unsigned char [28];
    file.read((char*) event_head, 28);
    if(file.tellg()==EOF){   
        return -1;
    }
    //The machine is supposed to be little endian.
    eventtype = (unsigned short) ((event_head[1]) << 8 | (event_head[0]));
    eventsource =  (unsigned short) ((event_head[3]) << 8 | (event_head[2]));
    eventsize =  (unsigned int) ((event_head[7]) << 24 | (event_head[6]<<16)|
                                 (event_head[5]) << 8  | (event_head[4]));
    eventoffset =  (unsigned int) ((event_head[11]) << 24 | (event_head[10]<<16)|
                                   (event_head[9])  << 8  | (event_head[8]));
    eventtsoverflow = (unsigned int) ((event_head[15]) << 24 | (event_head[14]<<16)|
                                      (event_head[13]) << 8  | (event_head[12]));
    eventcapacity = (unsigned int) ((event_head[19]) << 24 | (event_head[18]<<16)|
                                    (event_head[17]) << 8  | (event_head[16]));
    eventnumber = (unsigned int) ((event_head[23]) << 24 | (event_head[22]<<16)|
                                  (event_head[21]) << 8  | (event_head[20]));
    eventvalid = (unsigned int) ((event_head[27]) << 24 | (event_head[26]<<16)|
                                 (event_head[25]) << 8  | (event_head[24]));
    
    unsigned char * data = new unsigned char [eventsize];
    
    
    if(eventtype == 1){
    unsigned int event_data;
        for(int i=0; i<eventcapacity;i++){
            file.read((char*) data, eventsize);
            event_data = (unsigned int) ((data[3]) << 24 | (data[2]<<16)|
                                         (data[1]) << 8  | (data[0]));
            timestamp.push_back((unsigned int) ((data[7]) << 24 | (data[6]<<16)|
                                         (data[5]) << 8  | (data[4])));
            x_addr.push_back((unsigned int)(event_data >> 17) & 0x00007FFF);
            y_addr.push_back((unsigned int)(event_data >> 2) & 0x00007FFF);
            polarity.push_back((unsigned short)(event_data >> 1) & 0x00000001);


        }
    }

    if(eventtype==0){ //special event, the class is not developed to deal with this type, here for debugging only.
    std::vector <unsigned int> spec_ts;
    std::vector <unsigned short> spec_type;
        for(int i=0; i<eventcapacity;i++){
            file.read((char*) data, eventsize);
            spec_ts.push_back((unsigned int) ((data[7]) << 24 | (data[6]<<16)|
                                              (data[5]) << 8  | (data[4])));
            spec_type.push_back((unsigned short)(data[0]));
        }
    
    return 0;

    }
    
}


DAVISFrames::DAVISFrames(std::string fn, int dimx, int dimy){
    filename=fn;
    xdim=dimx;
    ydim=dimy;
    int eof_flag = 0;
    std::ifstream aedat_file;
    aedat_file.open(filename);
    if(!aedat_file.is_open())
    {
        std::cout<<"Huston we have a problem! File not found or inaccessible"<<std::endl;
    }
    else{
        skip_header(aedat_file);
        while(!(eof_flag)){
            eof_flag = read_frames(aedat_file, xdim, ydim, frames, start_ts, end_ts);
        }

    }
    aedat_file.close();

}

DAVISEvents::DAVISEvents(std::string fn){
    filename=fn;
    int eof_flag = 0;
    std::ifstream aedat_file;
    aedat_file.open(filename);
    if(!aedat_file.is_open())
    {
        std::cout<<"Huston we have a problem! File not found or inaccessible"<<std::endl;
    }
    else{
        skip_header(aedat_file);
        while(!(eof_flag)){
            eof_flag = read_events(aedat_file, polarity, timestamp, x_addr, y_addr);
        }

    }
    aedat_file.close();

}
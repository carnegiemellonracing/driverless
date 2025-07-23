setupHeaders(
    HEADERS
        BagStreamReaderFactoryExtension.hpp 
        BagStreamWriterFactoryExtension.hpp 
        IdcStreamReaderFactoryExtension.hpp 
        IdcStreamWriterFactoryExtension.hpp 

        TcpNetworkInterfaceFactoryExtension.hpp
        UdpNetworkInterfaceFactoryExtension.hpp
        
    GROUP extension\\io
    TARGET sdk-data
)
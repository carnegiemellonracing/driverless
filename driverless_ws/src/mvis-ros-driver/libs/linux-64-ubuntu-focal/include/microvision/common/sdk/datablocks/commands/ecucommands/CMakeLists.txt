setupHeaders(
    HEADERS
        AppBaseStatusDefinitions.hpp
        CommandEcuAppBaseCtrl2010Exporter2010.hpp
        CommandEcuAppBaseCtrl2010Importer2010.hpp
        CommandEcuAppBaseCtrlC.hpp
        CommandEcuAppBaseStatus2010Exporter2010.hpp
        CommandEcuAppBaseStatus2010Importer2010.hpp
        CommandEcuAppBaseStatusC.hpp
        CommandEcuSetFilter2010Exporter2010.hpp
        CommandEcuSetFilter2010Importer2010.hpp
        CommandEcuSetFilterC.hpp

        EcuCommand.hpp
        EcuCommandC.hpp

        ecucommands.hpp

        EmptyEcuCommandReply.hpp
        ReplyEcuAppBaseStatus.hpp
        ReplyEcuSetFilter.hpp

    GROUP datablocks\\commands\\ecucommands
    TARGET sdk-core
)


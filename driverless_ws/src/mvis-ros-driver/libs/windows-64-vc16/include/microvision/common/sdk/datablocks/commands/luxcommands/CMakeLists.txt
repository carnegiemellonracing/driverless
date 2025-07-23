setupHeaders(
    HEADERS
        CommandLuxGetParameter2010Importer2010.hpp
        CommandLuxGetParameter2010Exporter2010.hpp
        CommandLuxGetParameterC.hpp

        CommandLuxGetStatus2010Importer2010.hpp
        CommandLuxGetStatusC.hpp

        CommandLuxReset2010Importer2010.hpp
        CommandLuxResetC.hpp

        CommandLuxResetToDefaultParameters2010Importer2010.hpp
        CommandLuxResetToDefaultParametersC.hpp

        CommandLuxSaveConfig2010Importer2010.hpp
        CommandLuxSaveConfigC.hpp

        CommandLuxSetNtpTimestampSync2010Exporter2010.hpp
        CommandLuxSetNtpTimestampSync2010Importer2010.hpp
        CommandLuxSetNtpTimestampSyncC.hpp

        CommandLuxSetParameter2010Exporter2010.hpp
        CommandLuxSetParameter2010Importer2010.hpp
        CommandLuxSetParameterC.hpp

        CommandLuxStartMeasure2010Importer2010.hpp
        CommandLuxStartMeasureC.hpp

        CommandLuxStopMeasure2010Importer2010.hpp
        CommandLuxStopMeasureC.hpp

        EmptyLuxCommandReply.hpp
        LuxCommand.hpp
        LuxCommandC.hpp

        luxcommands.hpp

        ReplyLuxGetParameter.hpp
        ReplyLuxGetStatus.hpp

    GROUP datablocks\\commands\\luxcommands
    TARGET sdk-core
)

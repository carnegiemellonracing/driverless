# ==============================================================================
# Maps the RelWithDebInfo Config to the Release config.
# CMake by default uses the Debug config when a target is build with RelWithDebInfo
# but normally the Release config shall be used in this case,
# This function ensures that the Release config is used for certain targets.
# Extend the list of targets in the function if more targets need to be supported.
# ==============================================================================
function(common_mapRelWithDebInfoConfig)
    
    set(targetsList 
        tinyxml2
        )

    foreach (currentTarget ${targetsList})
        if (TARGET ${currentTarget})
            get_target_property(importedConfigs "${currentTarget}" IMPORTED_CONFIGURATIONS)
            
            if (NOT "RELWITHDEBINFO" IN_LIST importedConfigs)
                set_target_properties(${currentTarget} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
            endif()
        endif()    
    endforeach()

endfunction()   

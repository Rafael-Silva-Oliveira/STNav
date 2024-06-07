
params.sample_sheet_file = "test.csv"
params.revision          = "master"
params.outdir            = "results"
params.genome            = "GRCh37"
params.work              = "work"

// Define the process
process RunPipeline {

    publishDir "${params.outdir}"

    // Define the input parameters for the process
    input: 
    path sample_sheet_file
    val revision
    val genome
    val work

    // Define the output files produced by the process
    output: 
    path "results"

    // Define the script to be executed by the process
    script:
    """
    mkdir -p results
    nextflow -log results/nextflow.log run nf-core/rnaseq \\
        -r ${revision} \\
        --input ${sample_sheet_file} \\
        --outdir results \\
        --genome ${genome} \\
        -profile docker \\
        -w ${work}
    """
}

// Define the workflow
workflow {
    ch_input = Channel.fromPath(params.sample_sheet_file, checkIfExists: true, type: "file")
    // Run the process in the workflow
    RunPipeline(ch_input, params.revision, params.genome, params.work)
}


import streamlit as st


# Define the options for each selectbox
pre_processing_options = ["CleanPrimer", "Array To Fasta", "RACER"]
error_correction_options = ["eMER", "CORAL", "ECHO"]
exon_detection_options = ["AUGUSTUS", "BS-exon"]
mapping_options = ["BWA", "Biskit+GPU"]
differential_expression_options = ["edgeR", "cuffdiff", "DEseq"]
united_table_options = ["GenesIsoform Tables", "Unite from pipelines"]

# Custom HTML and CSS for styling and arrows
custom_css = """
<style>
.container {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.row {
    display: flex;
    align-items: center;
}
.arrow {
    width: 50px;
    height: 2px;
    background-color: black;
    margin: 10px;
}
.selectbox-container {
    margin: 20px;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

st.title("RNA-SEQ/CHIP Workflow")


# Layout the selectboxes with arrows
def create_workflow_step(label, options):
    st.markdown(
        f"<div class='row'><div class='selectbox-container'>{label}</div></div>",
        unsafe_allow_html=True,
    )
    selected_option = st.selectbox("", options, key=label)
    st.markdown("<div class='arrow'></div>", unsafe_allow_html=True)
    return selected_option


st.markdown("<div class='container'>", unsafe_allow_html=True)

pre_processing = create_workflow_step("Pre-Processing", pre_processing_options)
error_correction = create_workflow_step("Error Correction", error_correction_options)
exon_detection = create_workflow_step("Exon Detection", exon_detection_options)
mapping = create_workflow_step("Mapping on Transcripts", mapping_options)
differential_expression = create_workflow_step(
    "Differential Expression", differential_expression_options
)
united_table = create_workflow_step("United Table", united_table_options)

st.markdown("</div>", unsafe_allow_html=True)

st.write("### Workflow Summary")
st.write(f"Pre-Processing: {pre_processing}")
st.write(f"Error Correction: {error_correction}")
st.write(f"Exon Detection: {exon_detection}")
st.write(f"Mapping on Transcripts: {mapping}")
st.write(f"Differential Expression: {differential_expression}")
st.write(f"United Table: {united_table}")

# You can add more logic and display elements as needed

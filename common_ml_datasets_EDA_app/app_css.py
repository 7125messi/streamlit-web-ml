import streamlit as st

def main():
	st.title("XXX")

	activity = ['Design','About',]
	choice = st.sidebar.selectbox("Select Activity",activity)

	if choice == 'Design':
		st.subheader("Design")

	if choice =="About":
		st.subheader("About")
		st.info("YDZHAO")
		st.text("SF")
		st.success("Built with Streamlit")



if __name__ == '__main__':
	main()

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import base64
from wordcloud import WordCloud


# Load the model
model = tf.keras.models.load_model('Model_CNN_New_Genre_New (1).h5')

# Define genre mapping
genre_mapping = {
    0: 'disco', 1: 'metal', 2: 'reggae', 3: 'blues', 4: 'rock',
    5: 'classical', 6: 'country', 7: 'hiphop', 8: 'jazz', 9: 'pop', 10: 'Mezwed'
}


def audio_segment(X, window_size=0.1, overlap=0.5):
    S_X = []
    S_y = []
    total_sample = X.shape[0]
    # print('Total samples:{0}'.format(total_sample))

    number_windows = int(total_sample * window_size)
    # print('Total windows:{0}'.format(number_windows))

    number_overlap = int(number_windows * overlap)
    # print('Total overlap:{0}'.format(number_overlap))

    for i in range(0, total_sample - number_windows + number_overlap, number_overlap):
        S_X.append(X[i:i + number_windows])

    return np.array(S_X)



def to_mfcc(X):
    D = lambda x: librosa.feature.mfcc(
        S=librosa.power_to_db(librosa.feature.melspectrogram(y=x, n_fft=1024, hop_length=512)), n_mfcc=13)[:, :,
                  np.newaxis]
    f = map(D, X)
    return np.array(list(f))

def prep(music_path):
    n_rates = 660000
    window_size=0.1
    overlap=0.5
    data1, sr = librosa.load(music_path)
    n=len(data1)//n_rates
    datas=[]
    for i in range(n):
        data = data1[i*n_rates:(i+1)*n_rates]
        datas.append(data)
    return datas

def pre(music_path):
    datas = prep(music_path)
    musicpred = []
    for i in datas:
        X1 = np.array(i)
        print(X1.shape)
        X2 = audio_segment(X1)
        X3 = to_mfcc(X2)
        X = np.squeeze(np.stack((X3,) * 3, -1))
        predicted_genre = np.argmax(model.predict(X), axis = 1)
        musicpred.append(predicted_genre)

    output_list = np.concatenate(musicpred).tolist()
    return output_list


def predict_genre_and_statistics(music_path):
    datas = prep(music_path)
    musicpred = []
    probabilities = []
    for i in datas:
        X1 = np.array(i)
        X2 = audio_segment(X1)
        X3 = to_mfcc(X2)
        X = np.squeeze(np.stack((X3,) * 3, -1))

        max_predictions = np.max(model.predict(X), axis=1)
        probabilities.append(max_predictions)

        predicted_genre = np.argmax(model.predict(X), axis=1)
        musicpred.append(predicted_genre)

    output_list = np.concatenate(musicpred).tolist()
    output_proba = np.concatenate(probabilities).tolist()
    pattern = 19
    indices_to_remove = [1, 3, 5, 7, 9, 11, 13, 15, 17]

    filtered_output_list = [value for index, value in enumerate(output_list) if
                            (index % pattern) not in indices_to_remove]
    filtered_probabilities = [value for index, value in enumerate(output_proba) if
                              (index % pattern) not in indices_to_remove]
    segment_times = [i * 3 for i in range(len(filtered_output_list))]

    genre_counts = Counter(filtered_output_list)
    most_common_element = genre_counts.most_common(1)[0][0]
    most_common_genre = genre_mapping.get(most_common_element, "Unknown")

    genre_counts = {genre_mapping[genre]: count for genre, count in genre_counts.items()}
    total_occurrences = sum(genre_counts.values())

    threshold = total_occurrences * 0.1
    filtered_genre_counts = {genre: count for genre, count in genre_counts.items() if count >= threshold}

    data = []
    for i in range(len(filtered_output_list)):
        genre = genre_mapping.get(filtered_output_list[i], 'Unknown')
        probability = filtered_probabilities[i]
        data.append({'Time': segment_times[i], 'Value': probability, 'Class': genre})
    data.append({'Time': len(filtered_output_list)*3, 'Value': probability, 'Class': genre})

    return most_common_genre, filtered_genre_counts, data



def create_genre_probability_plot(data):
    fig, ax = plt.subplots(figsize=(12, 8))

    class_colors = {'disco': 'red', 'metal': 'green', 'reggae': 'blue', 'blues': 'purple', 'rock': 'orange',
                    'classical': 'pink', 'country': 'brown', 'hiphop': 'gray', 'jazz': 'cyan', 'pop': 'lime',
                    'Mezwed': 'black'}

    last_class = None
    last_point = None
    data[0]["Value"] = 0
    data[-1]["Value"] = 0
    data[-1]["Class"] = data[-2]["Class"]
    data[0]["Class"] = data[1]["Class"]

    for i in range(len(data) - 1):
        current_point = data[i]
        next_point = data[i + 1]
        next_class = next_point['Class']

        plt.plot([current_point['Time'], next_point['Time']],
                 [current_point['Value'], next_point['Value']],
                 color=class_colors.get(next_point['Class'], 'black'),
                 linewidth=2, markersize=8, linestyle='-', label=current_point['Class'])

        last_class = current_point['Class']

        plt.fill_between([current_point['Time'], next_point['Time']],
                         [current_point['Value'], next_point['Value']],
                         color=class_colors.get(next_point['Class'], 'black'), alpha=0.2)

    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title('Segment Distribution', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    ax1 = plt.gca()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    last_class = None
    last_point = None
    i = 0

    while i < len(data) - 1:
        current_point = data[i]
        next_point = data[i + 1]
        next_class = next_point['Class']

        if last_point is None:
            start_point = data[0]

        while current_point['Class'] == next_class and i < len(data) - 2:
            i += 1
            current_point = data[i]
            next_point = data[i + 1]
            next_class = next_point['Class']

        class_name = current_point['Class']
        x_position = (current_point['Time'] + start_point['Time']) / 2
        y_position = (current_point['Value'] + start_point['Value']) / 4
        plt.text(x_position, y_position, class_name, color='black',
                 ha='center', va='center', fontsize=12, weight='bold')

        last_class = current_point['Class']
        last_point = current_point
        start_point = current_point
        i += 1

    set_transparent_background(fig, ax)

    st.pyplot(plt)

def set_transparent_background(fig, ax):
    fig.patch.set_alpha(0.5)
    ax.patch.set_alpha(0)
def set_background_color(fig, ax, color):
    fig.patch.set_facecolor(color)
    ax.set_facecolor(color)





def create_genre_wordcloud(genre_distribution):
    wordcloud = WordCloud(width=800, height=400, background_color=None,colormap='Blues').generate_from_frequencies(genre_distribution)
    fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(10, 6))
    ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
    ax_wordcloud.axis('off')
    ax_wordcloud.set_title("Word Cloud for Genre Distribution")

    ax_wordcloud.set_facecolor((0, 0, 0, 0.5))  # Set a slightly transparent black background
    plt.gca().set_facecolor((0, 0, 0, 0.1))  # Set a slightly transparent black background for the entire plot

    return fig_wordcloud


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )




def main():
    st.set_page_config(
        page_title="Music Genre Classifier",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    set_bg_hack('a7a9948b-19de-4de3-a287-704db5639350.jpg')

    # App header
    st.title("Music Genre Classifier")

    # Sidebar navigation
    with open("c1311dc6-0a08-4774-b66c-4499017024b3.png", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

        st.sidebar.markdown(
            f"""
            <div style="display:table;margin-top:-20%;margin-left:5%;">
                <img src="data:image/png;base64,{data}" width="250" height="150">
            </div>
            """,
            unsafe_allow_html=True,
        )

    page = st.sidebar.radio("Go to", ("Upload Music", "Analyze Music"))

    if page == "Upload Music":
        upload_page()
    elif page == "Analyze Music":
        analyze_page()

def upload_page():
    st.title("Upload Music")
    uploaded_file = st.file_uploader("Upload a music file", type=["mp3", "wav"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        # Check if the "Analyze Music" button was clicked
        if st.button("Analyze Music", key="analyze_btn"):
            st.session_state.uploaded_file = uploaded_file
            st.write("Your analyze page is ready. Navigate to it through the menu bar on the left.")
            st.stop()  # Stop rendering after displaying the message





def analyze_page():
    st.title("Analyze Music")
    uploaded_file = st.session_state.uploaded_file

    if uploaded_file:
        most_common_genre, genre_distribution,datas1 = predict_genre_and_statistics(uploaded_file)

        st.subheader("Genre Insights")
        st.write(f"The most common genre is: **{most_common_genre}**")

        # Create columns for the plots
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Genre Histogram")
            st.pyplot(create_genre_histogram_plot(genre_distribution))

        with col2:
            st.subheader("Genre Probability Plot")
            create_genre_probability_plot(datas1)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Genre Distribution")
            st.pyplot(create_genre_pie_chart(genre_distribution))

        with col4:
            st.subheader("Genre Word Cloud")
            st.pyplot(create_genre_wordcloud(genre_distribution))







def visualize_genre_distribution(genre_distribution):
    st.pyplot(create_genre_pie_chart(genre_distribution))

def visualize_create_genre_probability_plot():
    st.pyplot(create_genre_probability_plot())

def visualize_genre_word_cloud(genre_distribution):
    st.pyplot(create_genre_wordcloud(genre_distribution))


# Create a pie chart for genre distribution
def create_genre_pie_chart(genre_distribution):
    labels = list(genre_distribution.keys())
    values = list(genre_distribution.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

    ax.set_title("Music Genre Distribution")

    set_transparent_background(fig, ax)

    return fig

def create_genre_histogram_plot(genre_distribution):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(genre_distribution.keys(), genre_distribution.values())
    ax.set_xlabel("Genres")
    ax.set_ylabel("Number of Segments")
    ax.set_title("Genre Distribution for the Entire Music File")
    ax.set_xticklabels(genre_distribution.keys(), rotation=45, ha="right")

    set_transparent_background(fig, ax)

    return fig
if __name__ == "__main__":
    main()

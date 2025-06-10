import json
import numpy as np
import pandas as pd
import random
import tkinter as tk
from tkinter import ttk, messagebox
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df= pd.read_excel('updated_4_5_315.xlsx', sheet_name='all_usable', engine='openpyxl')
df_user = pd.read_excel('updated_4_5_315.xlsx', sheet_name='user_friendly', engine='openpyxl')
CIP_MAP = {
    'CIP01BACHL': "Agriculture, Agriculture Operations, And Related Sciences",
    'CIP03BACHL': "Natural Resources And Conservation",
    'CIP04BACHL': "Architecture And Related Services",
    'CIP05BACHL': "Area, Ethnic, Cultural, Gender, And Group Studies",
    'CIP09BACHL': "Communication, Journalism, And Related Programs",
    'CIP10BACHL': "Communications Technologies/Technicians And Support Services",
    'CIP11BACHL': "Computer And Information Sciences And Support Services",
    'CIP12BACHL': "Personal And Culinary Services",
    'CIP13BACHL': "Education",
    'CIP14BACHL': "Engineering",
    'CIP15BACHL': "Engineering Technologies And Engineering-Related Fields",
    'CIP16BACHL': "Foreign Languages, Literatures, And Linguistics",
    'CIP19BACHL': "Family And Consumer Sciences/Human Sciences",
    'CIP22BACHL': "Legal Professions And Studies",
    'CIP23BACHL': "English Language And Literature/Letters",
    'CIP24BACHL': "Liberal Arts And Sciences, General Studies And Humanities",
    'CIP25BACHL': "Library Science",
    'CIP26BACHL': "Biological And Biomedical Sciences",
    'CIP27BACHL': "Mathematics And Statistics",
    'CIP29BACHL': "Military Technologies And Applied Sciences",
    'CIP30BACHL': "Multi/Interdisciplinary Studies",
    'CIP31BACHL': "Parks, Recreation, Leisure, And Fitness Studies",
    'CIP38BACHL': "Philosophy And Religious Studies",
    'CIP39BACHL': "Theology And Religious Vocations",
    'CIP40BACHL': "Physical Sciences",
    'CIP41BACHL': "Science Technologies/Technicians",
    'CIP42BACHL': "Psychology",
    'CIP43BACHL': "Homeland Security, Law Enforcement, Firefighting And Related Protective Services",
    'CIP44BACHL': "Public Administration And Social Service Professions",
    'CIP45BACHL': "Social Sciences",
    'CIP46BACHL': "Construction Trades",
    'CIP47BACHL': "Mechanic And Repair Technologies/Technicians",
    'CIP48BACHL': "Precision Production",
    'CIP49BACHL': "Transportation And Materials Moving",
    'CIP50BACHL': "Visual And Performing Arts",
    'CIP51BACHL': "Health Professions And Related Programs",
    'CIP52BACHL': "Business, Management, Marketing, And Related Support Services",
    'CIP54BACHL': "History"
}


class University_Recommender:
    def __init__(self, dataframe):
        self.data=dataframe
        self.variables=dataframe.shape[1]
        self.all_vectors=[]
        self.clusters={}
        self.centroids=[]
        self.current={}
        self.previous={}

    def data_vectors(self):
        self.all_vectors=[]
        for i in range(len(self.data)):
            vector=self.data.iloc[i].tolist()
            #norm=self.normalize_majors(vector)
            self.all_vectors.append((i,vector)) #map index excel ki file se aakhir mai
    
    def distance_normal(self,vector_1,vector_2):
        result=0
        for i in range(len(vector_1)):
            result+=(vector_1[i]-vector_2[i])**2 #baad mai include other formula for major ka variable
        return result
    
    def hamming_and_euclidean(self,vector_1,vector_2):
        euclidean=self.distance_normal(vector_1[0:5],vector_2[0:5])
        hamming=0
        for i in range(5,len(vector_1)):
            if(vector_1[i]!=vector_2[i]):
                hamming+=1
        return hamming+ euclidean
    
    def cosine_similarity(self,vector_1,vector_2):
        result=0
        norm_1=0
        norm_2=0
        for i in range(len(vector_1)):
            result+=(vector_1[i]*vector_2[i])
            norm_1+=vector_1[i]**2
            norm_2+=vector_2[i]**2
        denom=(norm_1**0.5)*(norm_2**0.5)
        return 1-(result/denom) if denom!=0 else 0
    
    def normalize_majors(self,vector):
        result=vector.copy()
        total=sum(result[9:self.variables])
        if(total==0):
            return result
        else:
            for i in range(9,self.variables):
                result[i]/=total
            return result
        
    def custom_distance(self,vector_1,vector_2):
        numerical_categorical=self.hamming_and_euclidean(vector_1[0:9],vector_2[0:9])
        flag=False
        for i in range(9,self.variables):
            if vector_1[i]==vector_2[i]==1:
                flag=True
                break
        return 1+(numerical_categorical) if flag==False else numerical_categorical
    
    def distance(self,vector_1,vector_2):
        return self.custom_distance(vector_1,vector_2)


    def centroids_selection(self, k): #using k++ initiallization
        index,uni=random.choice(self.all_vectors)
        self.centroids.append(uni)
        while len(self.centroids)<k:
            distance=[]
            for index, uni in self.all_vectors:
                dist_to_centroids=[]
                for centroid in self.centroids:
                    dist_to_centroids.append(self.distance(uni,centroid))
                distance.append(min(dist_to_centroids))
            temp=sum(distance)
            #print(temp)
            for i in range(len(distance)):
                distance[i]/=temp
            self.centroids.append(random.choices([uni for _,uni in self.all_vectors],weights=distance,k=1)[0])

    def new_clusters(self):
        for key in self.clusters:
            self.clusters[key].clear()
        for index, uni in self.all_vectors:
            distance=[]
            for centroid in self.centroids:
                distance.append(self.distance(uni,centroid))
            self.clusters[distance.index(min(distance))].append((index, uni))
            self.current[distance.index(min(distance))].append(index)
                
    def update_centroids(self):
        for i in range(len(self.clusters)):
            vector=[0]*self.variables
            if(len(self.clusters[i]))==0:
                continue
            for index, uni in self.clusters[i]:
                for k in range(self.variables): #tackle case for 0 elements aik cluster mai
                    vector[k]+=uni[k]
            for j in range(self.variables):
                vector[j]/=len(self.clusters[i])
                if j>=5:
                    vector[j]=round(vector[j])
            self.centroids[i]=vector

    def cluster_change(self): # check for when to terminate the loop
        changes=0
        for i in self.current:
            current_temp=set(self.current[i])
            previous_temp=set(self.previous[i])
            changes+=len(current_temp.symmetric_difference(previous_temp))
            self.previous[i]=list(current_temp)
            self.current[i]=[]
        return changes//2
    
    def k_means(self,k):
        print("satrted k-means")
        self.clusters={}
        self.previous={}
        self.current={}
        self.centroids=[]
        self.all_vectors=[]
        for i in range(k):
            self.clusters[i]=[]
            self.previous[i]=[]
            self.current[i]=[]
        flag=True
        self.data_vectors()
        self.centroids_selection(k)
        iteration=0
        while(flag or self.cluster_change()>10):
            iteration+=1
            print("changing clusters")
            flag=False
            self.new_clusters()
            self.update_centroids()
            if iteration>100:
                break

    """def k_clusters(self): #how many clusters to make
        ..."""
    
    def WCSS(self):
        result=0
        for cluster in self.clusters:
            for index,vector in self.clusters[cluster]:
                result+=self.distance(vector,self.centroids[cluster])
        return result
    
    def elbow_method(self):
        random.seed(1)
        wcss=[]
        k=[]
        current_k=0
        previous_k=0
        decrease=100
        for i in range(5,20):
            self.k_means(i)
            current_k=self.WCSS()
            k.append(i)
            wcss.append(current_k)
            if previous_k != 0:
                decrease=((previous_k-current_k)/previous_k)*100
                if decrease<10:
                    break
            previous_k=current_k
        
        plt.plot(k, wcss, marker='o')
        plt.yscale('log')  # optional
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS (log scale)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.show()

        return k[-1]

    def elbow_method_clean(self,):
        use_columns = ['SAT_NORM','ACT_NORM','TUTION_NORM','ADMR_NORM','COMPR_NORM']
        X = self.data[use_columns].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        wcss = []
        k_vals = range(2, 11)
        for k in k_vals:
            centroids = random.sample(list(X_scaled), k)
            for _ in range(10):
                clusters = [[] for _ in range(k)]
                for point in X_scaled:
                    distances = [np.linalg.norm(point - c) for c in centroids]
                    clusters[np.argmin(distances)].append(point)
                for i in range(k):
                    if clusters[i]:
                        centroids[i] = np.mean(clusters[i], axis=0)
            total_wcss = 0
            for i in range(k):
                total_wcss += sum(np.linalg.norm(p - centroids[i])**2 for p in clusters[i])
            wcss.append(total_wcss)
        plt.plot(k_vals, wcss, marker='o')
        plt.title('Clean Elbow Method (Euclidean Only)')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.grid(True)
        plt.show()

    
    def user_pref(self,user_input):
        """k=self.elbow_method()
        print(k)"""
        print("hello")
        self.k_means(7)
        print("K-means finished")
        mapped=(0,float('inf'))
        result=[]
        vector=user_input
        #vector=self.normalize_majors(user_input)
        for centroid in self.centroids:
            temp=self.distance(centroid,vector)
            if temp<mapped[1]:
                mapped=(self.centroids.index(centroid),temp)
        distance=[] 
        for index, uni in self.clusters[mapped[0]]:
            distance.append((self.distance(vector,uni),(index,uni)))
        if len(distance)<12:
            recommend=len(distance)
        else:
            recommend=12
        distance.sort(key=lambda x:x[0])
        result=distance[:recommend]
        return result
        
    def save_clusters_to_txt(self, filename="clusters_output.txt"):
        with open(filename, "w") as file:
            for i, centroid in enumerate(self.centroids):
                file.write(f"Cluster {i}:\n")
                file.write(f"Centroid: {centroid}\n")
                file.write("Members (index in original data):\n")
                for index, _ in self.clusters[i]:
                    file.write(f"{index}\n")
                file.write("\n" + "-"*50 + "\n\n")
        print(f"Clusters and centroids saved to {filename}")



class UniversityRecommenderGUI:
    def __init__(self, master, recommender, dataframe, user_friendly_df):
        self.master = master
        master.title("University Recommender GUI")

        self.rec = recommender
        self.df = dataframe               # original clustering data (all_usable)
        self.df_user = user_friendly_df   # user-friendly display data (user_friendly)
        self.cip_cols = [c for c in self.df.columns if c.startswith('CIP')]

        frm = ttk.Frame(master, padding=10)
        frm.grid(row=0, column=0, sticky='nw')

        # Numeric entries
        self.num_vars = {}
        row = 0
        for col in ['SAT_NORM','ACT_NORM','TUTION_NORM','ADMR_NORM','COMPR_NORM']:
            ttk.Label(frm, text=col).grid(row=row, column=0, sticky='e')
            ent = ttk.Entry(frm, width=12)
            ent.grid(row=row, column=1, sticky='w')
            self.num_vars[col] = ent
            row += 1

        # REGION dropdown
        ttk.Label(frm, text='REGION').grid(row=row, column=0, sticky='e')
        self.region_var = tk.StringVar()
        cb_region = ttk.Combobox(frm, textvariable=self.region_var, state='readonly', width=15)
        cb_region['values'] = ['US Service Schools','New England','Mid East','Great Lakes','Plains','Southeast','Souhtwest','Rocky Mountains','Far West','Outlying Area']
        cb_region.grid(row=row, column=1, sticky='w')
        row += 1

        # Gender Preference dropdown
        ttk.Label(frm, text='Gender Preference').grid(row=row, column=0, sticky='e')
        self.gender_var = tk.StringVar()
        cb_gender = ttk.Combobox(frm, textvariable=self.gender_var, state='readonly', width=15)
        cb_gender['values'] = ['Men-only', 'Women-only', 'Co-education']
        cb_gender.grid(row=row, column=1, sticky='w')
        row += 1

        # Public dropdown
        ttk.Label(frm, text='Public').grid(row=row, column=0, sticky='e')
        self.public_var = tk.StringVar()
        cb_pub = ttk.Combobox(frm, textvariable=self.public_var, state='readonly', width=10)
        cb_pub['values'] = ['private','public']
        cb_pub.grid(row=row, column=1, sticky='w')
        row += 1

        # Majors checkboxes
        ttk.Label(frm, text='Select Majors').grid(row=row, column=0, columnspan=2, sticky='w')
        row += 1
        majfrm = ttk.Frame(frm)
        majfrm.grid(row=row, column=0, columnspan=2, sticky='w')
        self.cip_vars = {}
        ccol, crow = 0, 0
        for cip in self.cip_cols:
            iv = tk.IntVar()
            # use friendly label instead of code
            chk = ttk.Checkbutton(majfrm, text=CIP_MAP[cip], variable=iv)
            chk.grid(row=crow, column=ccol, sticky='w')
            self.cip_vars[cip] = iv
            ccol += 1
            if ccol % 4 == 0:   # 4 columns per row for readability
                ccol = 0
                crow += 1
        row += 1

        # Show top 5 button
        btn = ttk.Button(frm, text='Show Top 5 Universities', command=self.show_top5)
        btn.grid(row=row, column=0, columnspan=2, pady=10)

        self.tree = None

    def normalize_userpref(self, vector):
        min=[820,14,2]
        max=[1550,35,228307]
        for i in range(3):
            temp=(vector[i]-min[i])/(max[i]-min[i])
            vector[i]=temp

    def show_top5(self):
            # Gather numeric inputs
        vec = [float(self.num_vars[c].get()) for c in ['SAT_NORM','ACT_NORM','TUTION_NORM','ADMR_NORM','COMPR_NORM']]
        try:
            if not (0 <= vec[0] <= 1600):
                raise ValueError("SAT must be between 0 and 1600")
            if not (0 <= vec[1] <= 36):
                raise ValueError("ACT must be between 0 and 36")
            if not (0 <= vec[2]):
                raise ValueError("Tuition must be non-negative")
            for x in vec[3:5]:
                if not (0 <= x <= 1):
                    raise ValueError("Rates must be between 0 and 1")
            self.normalize_userpref(vec)
            # REGION mapping (text → code)
            region_mapping = {
                'US Service Schools': 0,
                'New England': 1,
                'Mid East': 2,
                'Great Lakes': 3,
                'Plains': 4,
                'Southeast': 5,
                'Southwest': 6,
                'Rocky Mountains': 7,
                'Far West': 8,
                'Outlying Area': 9
            }
            selected_region = self.region_var.get()
            vec.append(region_mapping[selected_region])

            # Gender mapping
            g = self.gender_var.get()
            if g == 'Men-only':
                vec += [1, 0]
            elif g == 'Women-only':
                vec += [0, 1]
            else:  # Co-education
                vec += [0, 0]

            # Public/Private mapping
            public_mapping = {
                'private': 0,
                'public': 1
            }
            selected_public = self.public_var.get()
            vec.append(public_mapping[selected_public])

            # Majors
            for cip in self.cip_cols:
                vec.append(self.cip_vars[cip].get())
            print(vec)
            # Get recommendations
            recs = self.rec.user_pref(vec)
            top_serials = [idx for _, (idx, _) in recs[:5]]

        except Exception as e:
            messagebox.showerror('Input Error', str(e))
            return

        # Prepare columns: all non-CIP columns from user-friendly df plus 'Majors'
        non_cip_cols = [c for c in self.df_user.columns if not c.startswith('CIP')]
        cols = non_cip_cols + ['Majors']

        if self.tree:
            self.tree.destroy()
        self.tree = ttk.Treeview(self.master, columns=cols, show='headings', height=5)
        style = ttk.Style()
        style.configure("Treeview", rowheight=60)

        # Setup columns headers and widths
        for c in cols:
            if c == 'Majors':
                width = 300
            elif c == 'University':
                width = 250
            else:
                width = 100
            self.tree.heading(c, text=c)
            self.tree.column(c, width=width, anchor='w')

        # Insert rows with majors line-by-line
        for serial in top_serials:
            idx = serial + 1   # map 0→row1, etc.
            row_data = [self.df_user.at[idx, c] for c in non_cip_cols]

            # build a newline-separated string of friendly names
            offered = [CIP_MAP[cip] for cip in self.cip_cols if self.df.at[idx, cip] == 1]
            #majors_text = '\n'.join(offered)
            majors_text = '\n'.join(offered)
            row_data.append(majors_text)

            self.tree.insert('', tk.END, values=row_data)

        self.tree.grid(row=1, column=0, columnspan=2, padx=10, pady=10)


            
test=University_Recommender(df)
"""my_vector=[0.164383561643836,0.19047619047619,0.043840476555485,0.6622,0.287,5,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]	
my_list=test.user_pref(my_vector)
test.save_clusters_to_txt()
print(len(my_list))
for i in range(len(my_list)):
    print(my_list[i])"""

root = tk.Tk()
app = UniversityRecommenderGUI(root, test, df, df_user)
root.mainloop()
    
    
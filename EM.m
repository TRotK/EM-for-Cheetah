%% EM for cheetah segmentation
clear;clc;

tic;

load('TrainingSamplesDCT_8_new.mat');
cheetah=imread('cheetah.bmp'); 

[m1,n1]=size(TrainsampleDCT_BG);
[m2,n2]=size(TrainsampleDCT_FG);
Py_c=m2/(m1+m2);
Py_g=m1/(m1+m2);

%% Do DCT and ZIG-ZAG to the test image
cheetah=im2double(cheetah);
[m,n]=size(cheetah);
A_zigzag=ones(m*n,64);
cheetah1=padarray(cheetah,[7,7],'symmetric','post');
a=1;
for i=1:m
    for j=1:n
    A_zigzag(a,:)=new_zigzag(dct2(cheetah1(i:i+7,j:j+7)));
    a=a+1;
    end
end

C=8;

%initial parameters
Pi_g=ones(1,C)/C;

d=64;
TS_g = TrainsampleDCT_BG(:,1:d);
TS_c = TrainsampleDCT_FG(:,1:d);

param1 = cell(C,2);
for c = 1:C
    param1{c,1} = randn(d,1);
    param1{c,2} = diag(3+rand(d,1));
end

h_g = zeros(size(TS_g,1),C);
nh_g = 0;

px_g = zeros(size(h_g,1),C);

%error = zeros(25,64);

NumberOfIter=10;

%for grass=1:1

    for it=1:NumberOfIter
    
        for k = 1:C
            px_g(:,k) = mvnpdf(TS_g,param1{k,1}',param1{k,2});
        end

        for i = 1:size(h_g,2)
            
            % E-step
            
            h_g(:,i) = mvnpdf(TS_g,param1{i,1}',param1{i,2})*Pi_g(i); 
            
            h_g(:,i) = h_g(:,i)./(px_g*Pi_g');   %h_ij EM2,Page24
            
            % M-step
            nh_g = sum(h_g(:,i));       
            Pi_g(i) = nh_g/size(h_g,1);    % Pi_j^(n+1)
            param1{i,1} = sum(repmat(h_g(:,i),1,d).*TS_g)'/nh_g;   % mu_j^(n+1)
            
            sigma_update = zeros(d,d);
            
            for k = 1:size(h_g,1)
                xi_muj = TS_g(k,:)-param1{i,1}';
                sigma_update = sigma_update+h_g(k,i)*diag(diag(xi_muj'*xi_muj));
            end
            
            param1{i,2} = sigma_update/nh_g;    % EM2,Page27
        end
    end
    
    
    pe=zeros(5,11);
    for chta=1:5
         
        param2 = cell(C,2);
            
        for i = 1:C
            
            param2{i,1}=i*chta*2;     % just to be random
            
            %SIGMA0 = [.9 .4; .4 .3];
            param2{i,2}=500+chta*eye(d);  % just to be random
            
            %param2{i,1} = randn(d,1);
            %param2{i,2} = diag(3+rand(d,1));
        end
        
        h_c = zeros(size(TS_c,1),C);
        
        nh_c = 0;
        
        Pi_c = ones(1,C)/C;
        
        px_c = zeros(size(h_c,1),C);
        
        for it=1:NumberOfIter

           % param2 = cell(C,2);
            
            for k = 1:C
                px_c(:,k) = mvnpdf(TS_c,param2{k,1}',param2{k,2});
            end

            for i = 1:size(h_c,2)

                % E-step

                h_c(:,i) = mvnpdf(TS_c,param2{i,1}',param2{i,2})*Pi_c(i); 

                h_c(:,i) = h_c(:,i)./(px_c*Pi_c');   %h_ij EM2,Page24

                % M-step
                nh_c = sum(h_c(:,i));       
                Pi_c(i) = nh_c/size(h_c,1);    % Pi_j^(n+1)
                param2{i,1} = sum(repmat(h_c(:,i),1,d).*TS_c)'/nh_c;   % mu_j^(n+1)

                sigma_update = zeros(d,d);

                for k = 1:size(h_c,1)
                    xi_muj = TS_c(k,:)-param2{i,1}';
                    sigma_update = sigma_update+h_c(k,i)*diag(diag(xi_muj'*xi_muj));
                end

                param2{i,2} = sigma_update/nh_c;    % EM2,Page27
            end
        end
       
        % mixture gaussian
        
        countd=0;
        
        for d1=[1,2,4,8,16,24,32,40,48,56,64];      % dimension of features for classification
        
            countd=countd+1;
            
            paramd_g=cell(C,2);
            paramd_c=cell(C,2);

            
            
            x10=A_zigzag(:,1:d1);
           

            Pcheetah_g=0;
            Pcheetah_c=0;
            
            
            for component=1:C

                paramd_g{component,1}=param1{component,1}(1:d1);
                paramd_g{component,2}=param1{component,2}(1:d1,1:d1);

                paramd_c{component,1}=param2{component,1}(1:d1);
                paramd_c{component,2}=param2{component,2}(1:d1,1:d1);

            %end


            %for component=1:C

                Pcheetah_g=Pcheetah_g+mvnpdf(x10,paramd_g{component,1}',paramd_g{component,2})*Pi_g(component);
                Pcheetah_c=Pcheetah_c+mvnpdf(x10,paramd_c{component,1}',paramd_c{component,2})*Pi_c(component);

            end

             A=cheetah;
             
            

            %figure;imshow(im2uint8(A1));

           
            
             for row=1:m 
                for col=1:n

                zi=(row-1)*n+col;        
                %x1=x10(zi,:);
                    
                    
                if Pcheetah_g(zi)*Py_g>Pcheetah_c(zi)*Py_c

                   A(row,col)=0;

                else A(row,col)=1;

                end
                
                end
             end
                
            %error
            cheetah_mask=imread('cheetah_mask.bmp');
            cheetah_mask=im2double(cheetah_mask);
            
            A1=A(1:m-7,1:n-7);
            A1=padarray(A1,[4,4],'pre');
            A1=padarray(A1,[3,3],'post');

            a1=0;b1=0;
            a2=0;b2=0;
            
            for row=1:m
                for col=1:n
                
                    if cheetah_mask(row,col)==1&&A1(row,col)==1
                        b1=b1+1;
                        a1=a1+1;
                    elseif cheetah_mask(row,col)==1
                        b1=b1+1;
                    end
                    
                    if cheetah_mask(row,col)==0&&A1(row,col)==1
                        b2=b2+1;
                        a2=a2+1;
                    elseif cheetah_mask(row,col)==0
                        b2=b2+1;
                    end   

                end
            end


            ecc=a1/b1;ecg=a2/b2;
            pe(chta,countd)=ecg*Py_g+(1-ecc)*Py_c;
        
        end
       
        %figure(561);imshow(im2uint8(A1));
    end

%end
    
%plot        
figure(61);

mk=['o','+','*','s','p'];

%ls=['-','--','-.','-','--'];

di=[1 2 4 8 16 24 32 40 48 56 64];

for i=1:5
    %for D=1:11        
    %plot(dimention(1:D),pe(se,1:D),color(se),'Marker',mk(se),'LineWidth',2);grid on;
    
    plot(di,pe(i,:),'Marker',mk(i),'LineWidth',.8);grid on;
    hold on;
    %end    
end
hold off;

toc;       
        
legend('FG_1','FG_2','FG_3','FG_4','FG_5','Location','northeast');
xlabel('Dimension');ylabel('Probability of error');        

package com.bird.servlet;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.util.*;

import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileUploadException;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.servlet.ServletFileUpload;

@WebServlet(name = "RegisterServlet")
public class RegisterServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        if(!ServletFileUpload.isMultipartContent(request)){
            throw new RuntimeException("当前请求不支持文件上传");
        }

        try {
            //创建一个FileItem工厂
            DiskFileItemFactory factory = new DiskFileItemFactory();
            //创建新的核心组件
            ServletFileUpload upload = new ServletFileUpload(factory);
            //解析请求,获取所有item
            List<FileItem> items = upload.parseRequest(request);
            for(FileItem item : items){
                if(item.isFormField()){//若item为普通表单项
                    String fieldName = item.getFieldName();//获取表单名称
                    String fieldValue = item.getString();//获取表单项的值
                }else{//若item为文件表单项
                    String fileName = item.getName();//获取 上传文件原始名称
                    InputStream is = item.getInputStream();//获取输入流，其中有上传文件的内容
                    String path = this.getServletContext().getRealPath("./classes");//创建目标文件，将来用于保存上传文件
                    File descFile = new File(path, fileName);
                    OutputStream os = new FileOutputStream(descFile);//创建文件输出流
                    int len = -1;
                    byte[] buf = new byte[1024];
                    while((len = is.read(buf)) != -1){
                        os.write(buf, 0, len);
                    }
                    //关闭流
                    os.close();
                    is.close();
                }
            }
        } catch (FileUploadException e) {
            e.printStackTrace();
        }
    }

}

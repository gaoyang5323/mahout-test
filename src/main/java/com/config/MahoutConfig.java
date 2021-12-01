package com.config;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel;
import org.apache.mahout.cf.taste.impl.model.jdbc.ReloadFromJDBCDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.web.context.annotation.RequestScope;

import javax.sql.DataSource;
import java.io.IOException;

@Configuration
public class MahoutConfig {

    @Autowired
    DataSource dataSource;

    @Bean(value = "mySQLDataModel")
    public DataModel getMySQLJDBCDataModel() throws TasteException {
        return new ReloadFromJDBCDataModel(new MySQLJDBCDataModel(dataSource,
                "rating","user_id","movie_id","rating", "timestamp"));
    }

    @Bean(value = "fileDataModel")
    public DataModel getDataModel() throws IOException {
        Resource resource = new ClassPathResource("mahout/ratings-1m.data");
        return new FileDataModel(resource.getFile());
    }
}
